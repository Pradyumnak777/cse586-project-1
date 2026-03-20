import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from vPoser_test import pose_decode
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from data_utils import Data_VAE_time

'''
VAE setup, for decoding for MPJPE loss
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
support_dir = 'VPoserModelFiles'
expr_dir = os.path.join(support_dir,'vposer_v2_05/') #'TRAINED_MODEL_DIRECTORY'
vp, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True,
                              comp_device=device)
vp = vp.to(device)

'''
body. model post decoding, to get the joints pos
'''
bm_fname =  os.path.join(support_dir,'smplx_neutral_model.npz')    #'PATH_TO_SMPLX_model.npz'  neutral smpl body model
from human_body_prior.body_model.body_model import BodyModel
bm = BodyModel(bm_fname=bm_fname).to(device)


from data_utils import Data_VAE
from model import NextLatentTransformer

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#seed setup for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_train_stats(train_latents: dict, max_frames_per_seq=2000):
    # fast, stable stats for D=32
    chunks = []
    for _, v in train_latents.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        chunks.append(v[:max_frames_per_seq].float())
    cat = torch.cat(chunks, dim=0)  # (N, 32)
    mean = cat.mean(dim=0)
    std = cat.std(dim=0).clamp_min(1e-6)
    return mean, std

def rollout_autoreg(model, init_context_norm, H, device):
    """
    init_context_norm: (context, 32) normalized
    returns: (H, 32) normalized
    """
    model.eval()
    ctx = init_context_norm.to(device)
    preds = []
    with torch.no_grad():
        for _ in range(H):
            y = model(ctx.unsqueeze(0)).squeeze(0)  # (32,)
            preds.append(y.detach().cpu())
            ctx = torch.cat([ctx[1:], y.unsqueeze(0)], dim=0)
    return torch.stack(preds, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 420
    set_seed(seed)
    print("Using device:", device)
    print("Using seed:", seed)

    #const stuff
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    fps = 30
    context_window = 2 #in seconds
    pred_window = 0.033 #in seconds. predict 0.033s into the future
    batch_size = 128    
    
    latents_path = os.path.join(BASE_DIR, "vposer_latents.pt")
    latents_dict = torch.load(latents_path, map_location="cpu")
    #latents seem to be loaded at this point..

    keys = list(latents_dict.keys())
    train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=420) #this isout train/test split

    train_latents = {k: latents_dict[k] for k in train_keys} #so this is 01_01...? (file name)
    test_latents  = {k: latents_dict[k] for k in test_keys} 

    # #params
    # window = 1 #predicting "immediate" frame, true autorgression? change to 5 to predict 5 frames COLLECTIVELY
    # context = 20  #looking at 20 frames/poses
    
    epochs = 10
    lr = 2e-4
    noise_std = 0.01 #why this? does it help?

    loader_gen = torch.Generator().manual_seed(seed)

    
    '''
    BELOW IS FOR LONG TIME HORIZON (predicint x seconds into the future instead of immediate next frame!)
    '''
    train_ds = Data_VAE_time(train_latents, window_sec=pred_window, context_sec=context_window, fps=fps)
    test_ds  = Data_VAE_time(test_latents,  window_sec=pred_window, context_sec=context_window, fps=fps)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, generator=loader_gen)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, generator=loader_gen)

    # sanity shapes
    x0, y0 = next(iter(train_loader))
    print("Batch shapes:", x0.shape, y0.shape)  # expect (B, 20, 32)-input and (B, 32)-ground truth

    # ---- normalization computed from TRAIN only ----
    mean, std = compute_train_stats(train_latents)
    mean = mean.to(device)
    std  = std.to(device)

    #model
    model = NextLatentTransformer(d_in=32, d_model=128, nhead=4, num_layers=3, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.L1Loss() #alternative is MSE

    for ep in range(1, epochs + 1):
        # train
        print(f"\n{'='*30} EPOCH {ep} {'='*30}", flush=True)
        print(f"--- Phase: Training ---", flush=True)
        model.train()
        total = 0.0
        train_latent_sum = 0.0
        train_mpjpe_sum = 0.0
        train_total_sum = 0.0
        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).float()

            # normalize here (no dataset edits)
            x = (x - mean) / std
            y = (y - mean) / std

            # noise on input context only (train)
            x = x + noise_std * torch.randn_like(x)

            pred = model(x) #input- [B, 20, 32]; output-[B, 1, 32]
            loss_latent = loss_fn(pred, y)
            
            '''
            Adding MPJPE loss below!!
            '''
            pred_latent = (pred * std) + mean #un normalizing
            gt_latent = (y * std) + mean
            #decode ts
            decoded_pred_pose = pose_decode(vp, pred_latent) 
            decoded_gt_pose = pose_decode(vp, gt_latent)
            pred_joints = bm(pose_body=decoded_pred_pose).Jtr[:, :23, :] 
            gt_joints = bm(pose_body=decoded_gt_pose).Jtr[:, :23, :]    
            loss_mpjpe = torch.mean(torch.norm(pred_joints - gt_joints, dim=-1))
            
            total_loss = loss_latent + (0.1 * loss_mpjpe) #CHANGE 0.1 to something else later??
            '''
            decoding after every pass seems stupid to me, to calc loss. (MPJPE)
            need to think of something better
            '''
            
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_latent_sum += loss_latent.item()
            train_mpjpe_sum += loss_mpjpe.item()
            train_total_sum += total_loss.item()

        avg_train_lat = train_latent_sum / len(train_loader)
        avg_train_mpj = train_mpjpe_sum / len(train_loader)
        avg_train_tot = train_total_sum / len(train_loader)
        print(f"Epoch {ep} TRAIN | Total: {avg_train_tot:.4f} | Latent (MAE): {avg_train_lat:.4f} | Physical (MPJPE): {avg_train_mpj:.4f}m", flush=True)
        
        # # test
        # print(f"--- Phase: Testing ---", flush=True)
        # model.eval()
        # test_latent_sum = 0.0
        # test_mpjpe_sum = 0.0
        # test_total_sum = 0.0

        # with torch.no_grad():
        #     for x, y in test_loader:
        #         x, y = x.to(device).float(), y.to(device).float()
                
        #         # Normalize
        #         x_norm = (x - mean) / std
        #         y_norm = (y - mean) / std
                
        #         # Predict
        #         pred_norm = model(x_norm)
                
        #         # 1. Latent MAE
        #         batch_loss_latent = loss_fn(pred_norm, y_norm)
        #         test_latent_sum += batch_loss_latent.item()
                
        #         # 2. MPJPE (un-normalize first)
        #         pred_latent = (pred_norm * std) + mean
        #         gt_latent = y # Ground truth is already un-normalized in the original dict
                
        #         decoded_pred = pose_decode(vp, pred_latent)
        #         decoded_gt = pose_decode(vp, gt_latent)
                
        #         # Get 3D joints for evaluation
        #         pj = bm(pose_body=decoded_pred).Jtr[:, :23, :]
        #         gj = bm(pose_body=decoded_gt).Jtr[:, :23, :]
                
        #         batch_mpjpe = torch.mean(torch.norm(pj - gj, dim=-1))
        #         test_mpjpe_sum += batch_mpjpe.item()
                
        #         batch_total_loss = batch_loss_latent + (0.1 * batch_mpjpe)
        #         test_total_sum += batch_total_loss.item()
            
            
        #     avg_test_lat = test_latent_sum / len(test_loader)
        #     avg_test_mpj = test_mpjpe_sum / len(test_loader)
        #     avg_test_tot = test_total_sum / len(test_loader)
        #     print(f"Epoch {ep} TEST  | Total: {avg_test_tot:.4f} | Latent (MAE): {avg_test_lat:.4f} | Physical (MPJPE): {avg_test_mpj:.4f}m", flush=True)
            
        save_dir = os.path.join(BASE_DIR, f"test_models_context_{context_window}_k_1")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"transformer_ep{ep}.pt"))

    
    # print("\n--- doing ROLLOUT ---")
    # model.eval()
    
    # # #fix: Calculate frame counts from seconds once
    # c_frames = int(round(context_window * fps)) # 2 * 30 = 60
    
    # eval_seconds = [1.0, 2.0, 3.0, 5.0]
    # horizons = [int(round(s * fps)) for s in eval_seconds]
    # horizon_mae = {h: [] for h in horizons}
    
    # with torch.no_grad():
    #     for key in test_keys:
    #         full_seq = test_latents[key].to(device).float()
            
    #         # #fix: Compare frames to frames, not frames to seconds
    #         if full_seq.shape[0] < (c_frames + 200):
    #             continue 
                
    #         # #fix: Use frame count for slicing
    #         init_context = full_seq[:c_frames] 
    #         init_context_norm = (init_context - mean) / std
            
    #         # This generates 200 frames of future motion
    #         preds_norm = rollout_autoreg(model, init_context_norm, H=200, device=device) 
            
    #         # #fix: Slicing ground truth starting from the end of context frames
    #         gt_rollout = full_seq[c_frames : c_frames + 200]
    #         gt_rollout_norm = ((gt_rollout - mean) / std).to(device)
            
    #         for h in horizons:
    #             if h <= len(preds_norm):
    #                 err = torch.mean(torch.abs(preds_norm[h-1].to(device) - gt_rollout_norm[h-1]))
    #                 horizon_mae[h].append(err.item())
    
    # print("\n Rollout Statistics (Normalized Latent MAE)")
    # print(f"{'Time':<20} | {'Frames':<10} | {'Mean Error':<15}")
    # print("-" * 55)
    # for h in horizons:
    #     avg_err = np.mean(horizon_mae[h])
    #     time_sec = h / 30.0 
    #     print(f"{time_sec:>4.1f} seconds {h:>10} frames {avg_err:>15.4f}")
    # print("===END!!===")   
    
    # '''
    # below doesnt make sense, just saving transfoermer model instead...
    # '''
    # # torch.save({
    # #     "context_norm": init_context_norm.detach().cpu(),
    # #     "pred_latents_norm": pred_latents_norm.cpu(),
    # #     "mean": mean.detach().cpu(),
    # #     "std": std.detach().cpu(),
    # #     "context_len": context,
    # #     "rollout_len": H
    # # }, os.path.join(BASE_DIR, "pred_latents.pt"))

    # # print("Saved pred_latents.pt (normalized preds + mean/std).")

if __name__ == "__main__":
    main()


#NEW STATS below
'''
Epoch 1 TRAIN | Total: 0.1329 | Latent (MAE): 0.1311 | Physical (MPJPE): 0.0181m
--- Phase: Testing ---
Epoch 1 TEST  | Total: 0.0499 | Latent (MAE): 0.0492 | Physical (MPJPE): 0.0066m

============================== EPOCH 2 ==============================
--- Phase: Training ---
Epoch 2 TRAIN | Total: 0.0566 | Latent (MAE): 0.0558 | Physical (MPJPE): 0.0073m
--- Phase: Testing ---
Epoch 2 TEST  | Total: 0.0335 | Latent (MAE): 0.0331 | Physical (MPJPE): 0.0041m

============================== EPOCH 3 ==============================
--- Phase: Training ---
Epoch 3 TRAIN | Total: 0.0405 | Latent (MAE): 0.0400 | Physical (MPJPE): 0.0050m
--- Phase: Testing ---
Epoch 3 TEST  | Total: 0.0290 | Latent (MAE): 0.0287 | Physical (MPJPE): 0.0034m

============================== EPOCH 4 ==============================
--- Phase: Training ---
Epoch 4 TRAIN | Total: 0.0335 | Latent (MAE): 0.0331 | Physical (MPJPE): 0.0039m
--- Phase: Testing ---
Epoch 4 TEST  | Total: 0.0254 | Latent (MAE): 0.0252 | Physical (MPJPE): 0.0028m

============================== EPOCH 5 ==============================
--- Phase: Training ---
Epoch 5 TRAIN | Total: 0.0300 | Latent (MAE): 0.0296 | Physical (MPJPE): 0.0034m
--- Phase: Testing ---
Epoch 5 TEST  | Total: 0.0230 | Latent (MAE): 0.0228 | Physical (MPJPE): 0.0023m

--- doing ROLLOUT ---

================ Rollout Statistics (Normalized Latent MAE) ================
Time                 | Frames     | Mean Error     
-------------------------------------------------------
 1.0 seconds         30 frames          0.6103
 2.0 seconds         60 frames          0.9272
 3.0 seconds         90 frames          1.0786
 5.0 seconds        150 frames          1.0639
============================================================================

'''


#OLD STATS BELOW!! (ARCHIVE)

"""
==========================
TESTING RESULTS SUMMARY
==========================

Setup:
- Latent dimension: 32 (VPoser latent space)
- Context length: 20 frames
- Prediction: autoregressive rollout for 200 steps
- Loss: SmoothL1 (Huber) in normalized latent space
- Device: MPS (Apple Silicon)

Training Convergence:
Epoch 1 train loss: 0.0226 | test loss: 0.0029
Epoch 2 train loss: 0.0036 | test loss: 0.0018
Epoch 3 train loss: 0.0019 | test loss: 0.0013
Epoch 4 train loss: 0.0012 | test loss: 0.0012
Epoch 5 train loss: 0.0010 | test loss: 0.0010

Observation:
- Train and test losses converge smoothly.
- No signs of overfitting (train ≈ test).
- Model stable under autoregressive rollout (no exploding values).

Autoregressive Stability Check:
- Rollout length: 200
- Predicted latent std ≈ 1.24
- Max absolute value ≈ 3.24
=> No divergence or instability observed.

Comparison vs Constant Velocity Baseline (latent MSE over rollout):

Sequence        Model MSE    Baseline MSE    Better?
-----------------------------------------------------
13_38_poses     1.526        8.267           Yes
02_01_poses     1.268        27.575          Yes
05_04_poses     1.907        0.388           No
13_40_poses     1.620        101.196         Yes
01_10_poses     1.052        26.303          Yes

Overall:
- Model outperforms constant-velocity baseline on 4/5 test sequences.
- Baseline wins only in nearly linear motion cases.
- Transformer better captures nonlinear motion dynamics.

Conclusion:
The autoregressive Transformer improves over simple motion baselines
in most cases and demonstrates stable long-horizon rollout behavior.

Next Step:
Decode predicted latents → SMPL pose → Joints → Evaluate MPJPE in joint space.
"""
import torch
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_utils import Data_VAE

# Use the loaders you already defined in vPoser_test.py
from vPoser_test import load_vposer, body_model_loading

# --- 1. BASELINE CALCULATIONS ---

def predict_zero_velocity(history, future_len):
    """Baseline 1: Just repeat the last known frame."""
    last_frame = history[:, -1, :] 
    return last_frame.unsqueeze(1).repeat(1, future_len, 1)

def predict_constant_velocity(history, future_len):
    """Baseline 2: Project movement based on the last two frames."""
    p_t = history[:, -1, :]
    p_t_1 = history[:, -2, :]
    vel = p_t - p_t_1
    
    preds = []
    curr = p_t
    for _ in range(future_len):
        curr = curr + vel
        preds.append(curr)
    return torch.stack(preds, dim=1)

def compute_mpjpe(pred_j, gt_j):
    """Mean Per Joint Position Error in Millimeters."""
    # Joint distance calculation
    dist = torch.norm(pred_j - gt_j, dim=-1) 
    return torch.mean(dist).item() * 1000.0 # Convert to mm

# --- 2. DATA LOADING ---

latents_dict = torch.load('vposer_latents.pt', weights_only=False) 

# **NEW**: Load your Transformer predictions!
transformer_preds = torch.load('pred_latents.pt', weights_only=False)
print(latents_dict.keys(),len(latents_dict),latents_dict) # Check keys to ensure they match test_latents

print(transformer_preds.keys(),len(transformer_preds),transformer_preds) # Check keys to ensure they match test_latents
k = list(latents_dict.keys())
train_poses, test_poses = train_test_split(k, test_size=0.2, random_state=420)

test_latents = {k: latents_dict[k] for k in test_poses}

# --- 3. THE "BETTER" TIER EVALUATION ---

def run_baseline_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- PATH SETUP CORRECTED ---
    bm_fname = 'VPoserModelFiles/smplx_neutral_model.npz' 
    expr_dir = 'VPoserModelFiles/vposer_v2_05/' 

    print("Loading Models for Evaluation...")
    vp, _ = load_vposer(expr_dir, device) 
    bm = body_model_loading(bm_fname, device)
    vp.eval()

    horizons = {'80ms': 10, '320ms': 38, '1000ms': 120}
    max_f = 120
    context = 10

    # **NEW**: Added a dictionary key for your Transformer results
    results = {"Zero-Vel": {h: [] for h in horizons}, 
               "Const-Vel": {h: [] for h in horizons},
               "Transformer": {h: [] for h in horizons}}

    with torch.no_grad():
        for name in list(test_latents.keys())[:10]: # Test on first 10 sequences
            seq = test_latents[name].to(device)
            if len(seq) < (context + max_f): continue
            
            past = seq[:context].unsqueeze(0)
            gt = seq[context : context + max_f].unsqueeze(0)

            # 1. Predict in Latent Space (32D)
            zv_p = predict_zero_velocity(past, max_f)
            cv_p = predict_constant_velocity(past, max_f)
            
            # **NEW**: Grab your Transformer's predicted latents for this sequence
            # Slicing to max_f to ensure it perfectly matches the ground truth length
            trans_p = transformer_preds[name].to(device)
            if trans_p.dim() == 2:
                trans_p = trans_p.unsqueeze(0) # Make sure shape is [1, frames, 32]
            trans_p = trans_p[:, :max_f, :]

            # 2. Decode Latents -> Pose Parameters -> 3D Joints
            def decode_to_j(latents):
                pose_body = vp.decode(latents)['pose_body'].reshape(-1, 63)
                return bm(pose_body=pose_body).Jtr

            gt_j = decode_to_j(gt)
            zv_j = decode_to_j(zv_p)
            cv_j = decode_to_j(cv_p)
            trans_j = decode_to_j(trans_p) # **NEW**: Decode Transformer latents

            # 3. Calculate Error per Horizon
            for h_n, h_f in horizons.items():
                results["Zero-Vel"][h_n].append(compute_mpjpe(zv_j[:h_f], gt_j[:h_f]))
                results["Const-Vel"][h_n].append(compute_mpjpe(cv_j[:h_f], gt_j[:h_f]))
                # **NEW**: Calculate MPJPE for Transformer
                results["Transformer"][h_n].append(compute_mpjpe(trans_j[:h_f], gt_j[:h_f]))

    print("\n--- SUNDAY PROGRESS REPORT RESULTS ---")
    for h in horizons:
        avg_zv = sum(results["Zero-Vel"][h]) / len(results["Zero-Vel"][h])
        avg_cv = sum(results["Const-Vel"][h]) / len(results["Const-Vel"][h])
        avg_tr = sum(results["Transformer"][h]) / len(results["Transformer"][h]) # **NEW**
        
        print(f"Horizon {h}: Zero-Vel: {avg_zv:.2f}mm | Const-Vel: {avg_cv:.2f}mm | Transformer: {avg_tr:.2f}mm")

if __name__ == "__main__":
    #run_baseline_benchmarks()
    pass
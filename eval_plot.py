import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from vPoser_test import pose_decode
from human_body_prior.tools.model_loader import load_model
from human_body_prior.body_model.body_model import BodyModel
from model import NextLatentTransformer
from human_body_prior.models.vposer_model import VPoser
import trimesh
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import matplotlib.pyplot as plt


#the loop that makes model predict the future 
'''
    ok so for rollout, its like autoregression. 
    [p1, p2,..,p20] : input
    p21 : output
    
    Now, instead of just using the groundtruth p21 to predict p22 like in testing, here p21 is p21*(model output previously)
    [p2, p3..,p21*] : input
    p22 : output
'''
def rollout_autoreg(model, init_context_norm, H, device):
    model.eval()
    ctx = init_context_norm.to(device)
    preds = []
    with torch.no_grad():
        for _ in range(H):
            y = model(ctx.unsqueeze(0)).squeeze(0) 
            preds.append(y.detach().cpu())
            ctx = torch.cat([ctx[1:], y.unsqueeze(0)], dim=0)
    return torch.stack(preds, dim=0)

#helper to compute stats
def compute_train_stats(train_latents: dict, max_frames_per_seq=2000):
    chunks = []
    for _, v in train_latents.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        chunks.append(v[:max_frames_per_seq].float())
    cat = torch.cat(chunks, dim=0) 
    mean = cat.mean(dim=0)
    std = cat.std(dim=0).clamp_min(1e-6)
    return mean, std

def plot_mpjpe_divergence(all_frame_stats, fps):
    #take mean error at each frame across all test sequences
    final_errors = {
        'trans': [np.mean(frame_list) for frame_list in all_frame_stats['trans']],
        'zv': [np.mean(frame_list) for frame_list in all_frame_stats['zv']],
        'cv': [np.mean(frame_list) for frame_list in all_frame_stats['cv']]
    }
    
    #build time axis in seconds
    num_frames = len(final_errors['trans'])
    time_steps = np.arange(1, num_frames + 1) / fps
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_steps, final_errors['trans'], label='transformer', color='hotpink', linewidth=2)
    
    #ts are baseline curves
    plt.plot(time_steps, final_errors['zv'], label='zero velocity', color='gray', linestyle='--')
    plt.plot(time_steps, final_errors['cv'], label='constant velocity', color='royalblue', linestyle=':')
    
    plt.title('mpjpe divergence over five seconds')
    plt.xlabel('time into the future (seconds)')
    plt.ylabel('mean error (meters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('eval_viz', exist_ok=True)
    plt.savefig('eval_viz/divergence_plot.png')
    print('divergence plot saved to eval_viz folder')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    #const
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    fps = 30 #subsampled from 120fps
    context_window = 2 #2 seconds of history (transformer was alos trained on ts config)
    c_frames = int(round(context_window * fps)) #60 frames (cuz we subsampled to 30 frames)
    
    #vposer for decoding latent output by transformer
    print("loading vposer and smpl...")
    support_dir = 'VPoserModelFiles'
    expr_dir = os.path.join(support_dir, 'vposer_v2_05/')
    vp, _ = load_model(expr_dir, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True, comp_device=device)
    vp = vp.to(device)
    
    bm_fname = os.path.join(support_dir, 'smplx_neutral_model.npz')
    bm = BodyModel(bm_fname=bm_fname).to(device)

    #load our latent data and split exactly like training
    latents_path = os.path.join(BASE_DIR, "vposer_latents.pt")
    latents_dict = torch.load(latents_path, map_location="cpu")
    
    keys = list(latents_dict.keys())
    #used 420 so the test set is exactly the same one we validated on
    train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=420) 

    train_latents = {k: latents_dict[k] for k in train_keys}
    test_latents  = {k: latents_dict[k] for k in test_keys} 

    #nned to un-normalize..
    '''
    seems like the predictions also need to be un-normed to calculate MPJPE. did this in training too..
    '''
    mean, std = compute_train_stats(train_latents)
    mean, std = mean.to(device), std.to(device)

    #load our trained transformer
    print("loading trained transformer weights...")
    model = NextLatentTransformer(d_in=32, d_model=128, nhead=4, num_layers=3, dropout=0.1).to(device)
    
    #load model
    model_path = os.path.join(BASE_DIR, "best_model/transformer_ep40.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #define the time horizons in pdf
    eval_seconds = [0.067, 0.333, 1.0, 2.0, 5.0] 
    horizons = [int(round(s * fps)) for s in eval_seconds] #in frames
    max_h = max(horizons) #we only need to rollout as far as our max horizon
    
    #dictionaries to store the mpjpe errors
    stats = {
        'trans': {h: [] for h in horizons}, #generated by transformer rollout!
        'zv':    {h: [] for h in horizons}, #zero velocity
        'cv':    {h: [] for h in horizons} #const. velocity
    }

    #track every single frame for the plot
    all_frame_stats = {
        'trans': [[] for _ in range(max_h)],
        'zv':    [[] for _ in range(max_h)],
        'cv':    [[] for _ in range(max_h)]
    }

    print("\nstarting combined evaluation... ")
    
    with torch.no_grad():
        for key in test_keys:
            full_seq = test_latents[key].to(device).float()
            
            if full_seq.shape[0] < (c_frames + max_h):
                continue 
                
            #get context
            init_context = full_seq[:c_frames] #c_frames is context, but in frames
            init_context_norm = (init_context - mean) / std
            
            #doing rollout once per sequence
            preds_norm = rollout_autoreg(model, init_context_norm, H=max_h, device=device) #rollout for 150 frames
            
            #get gt (already un-normalized)
            gt_rollout = full_seq[c_frames : c_frames + max_h]
            
            last_frame = init_context[-1] #60th frame in this case
            penultimate_frame = init_context[-2] #59th frame
            velocity = last_frame - penultimate_frame
            
            #loop through every frame to save both table and plot data
            for h in range(1, max_h + 1):
                trans_latent = (preds_norm[h-1].to(device) * std) + mean #un_normed transformer output at this 'h' step
                
                zv_latent = last_frame
                cv_latent = last_frame + (h * velocity)
                gt_latent = gt_rollout[h-1]
                
                trans_pose = pose_decode(vp, trans_latent.unsqueeze(0))
                zv_pose    = pose_decode(vp, zv_latent.unsqueeze(0))
                cv_pose    = pose_decode(vp, cv_latent.unsqueeze(0))
                gt_pose    = pose_decode(vp, gt_latent.unsqueeze(0))
                
                trans_joints = bm(pose_body=trans_pose).Jtr[:, :23, :]
                zv_joints    = bm(pose_body=zv_pose).Jtr[:, :23, :]
                cv_joints    = bm(pose_body=cv_pose).Jtr[:, :23, :]
                gt_joints    = bm(pose_body=gt_pose).Jtr[:, :23, :]
                
                err_trans = torch.mean(torch.norm(trans_joints - gt_joints, dim=-1))
                err_zv    = torch.mean(torch.norm(zv_joints - gt_joints, dim=-1))
                err_cv    = torch.mean(torch.norm(cv_joints - gt_joints, dim=-1))
                
                all_frame_stats['trans'][h-1].append(err_trans.item())
                all_frame_stats['zv'][h-1].append(err_zv.item())
                all_frame_stats['cv'][h-1].append(err_cv.item())

                if h in horizons:
                    stats['trans'][h].append(err_trans.item())
                    stats['zv'][h].append(err_zv.item())
                    stats['cv'][h].append(err_cv.item())
                

    print("\n Evaluation of MPJPE error (in meters)")
    print(f"{'Time (s)':<10} | {'Transformer':<15} | {'Zero-Vel':<15} | {'Const-Vel':<15}")
    print("-" * 65)
    
    for h in horizons:
        t = h / fps
        m_trans = np.mean(stats['trans'][h])
        m_zv    = np.mean(stats['zv'][h])
        m_cv    = np.mean(stats['cv'][h])
        
        #adding star if it beats the baselines
        trans_str = f"{m_trans:>15.4f}"
        if m_trans < m_zv and m_trans < m_cv:
            trans_str = f"*{m_trans:>13.4f}*" #adding asterisks to highlight the winner
            
        print(f"{t:>8.3f}s | {trans_str} | {m_zv:>15.4f} | {m_cv:>15.4f}")
    print("--END--")
    
    #generate the plot after everything is done
    plot_mpjpe_divergence(all_frame_stats, fps)
    
    #extract faces from the body model for trimesh
    faces = c2c(bm.f)
    
    seq_scores = {}
    for key in test_keys:
        #average the error across all time horizons for this specific key
        avg_err = np.mean([np.mean(stats['trans'][h_idx]) for h_idx in horizons])
        seq_scores[key] = avg_err
        
    sorted_keys = sorted(seq_scores, key=seq_scores.get)
    best_key = sorted_keys[0]
    worst_key = sorted_keys[-1]

    print(f"best: {best_key} | worst: {worst_key}")
    
    h_eval = 30 #looking at body poses at the 1st second
    
    viz_stuff = {
        'best': {
            'key': best_key,
            'gt_latent': test_latents[best_key][60 + h_eval], # 1.0s mark, first 60 is context(set to 2 sec)
            'init_context': test_latents[best_key][:60]
        },
        'worst': {
            'key': worst_key,
            'gt_latent': test_latents[worst_key][60 + h_eval],
            'init_context': test_latents[worst_key][:60]
        },
        'stats': {
            'mean': mean.cpu(),
            'std': std.cpu()
        }
    }

    torch.save(viz_stuff, 'viz.pt')
    print("viz stuff saved for portability..")
    
if __name__ == "__main__":
    main()
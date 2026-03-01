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

# # --- 2. DATA LOADING ---

# latents_dict = torch.load('vposer_latents.pt')
# k = list(latents_dict.keys())
# train_poses, test_poses = train_test_split(k, test_size=0.2, random_state=420)

# test_latents = {k: latents_dict[k] for k in test_poses}

# # --- 3. THE "BETTER" TIER EVALUATION ---

# def run_baseline_benchmarks():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # PATH SETUP
#     # Point this to the .npz you have in your repo
#     bm_fname = 'smplx_neutral_model.npz' 
    
#     # Point this to wherever the actual VPoser model weights are
#     # If you don't have this folder, you will need to download the 'vposer_v2_05' folder
#     expr_dir = 'VPoserModelFiles/vposer_v2_05/' 

#     print("Loading Models for Evaluation...")
#     vp, _ = load_vposer(expr_dir, device)
#     bm = body_model_loading(bm_fname, device)
#     vp.eval()

#     # Define the 3 horizons required by the project
#     horizons = {'80ms': 10, '320ms': 38, '1000ms': 120}
#     max_f = 120
#     context = 10

#     results = {"Zero-Vel": {h: [] for h in horizons}, "Const-Vel": {h: [] for h in horizons}}

#     with torch.no_grad():
#         for name in list(test_latents.keys())[:10]: # Test on first 10 sequences
#             seq = test_latents[name].to(device)
#             if len(seq) < (context + max_f): continue
            
#             past = seq[:context].unsqueeze(0)
#             gt = seq[context : context + max_f].unsqueeze(0)

#             # 1. Predict in Latent Space
#             zv_p = predict_zero_velocity(past, max_f)
#             cv_p = predict_constant_velocity(past, max_f)

#             # 2. Decode to 3D Joints (The "Decoding" part you asked about)
#             def decode_to_j(latents):
#                 # Turn 32D -> 63D rotations
#                 pose_body = vp.decode(latents)['pose_body'].reshape(-1, 63)
#                 # Turn 63D rotations -> 3D XYZ Joints using the .npz model
#                 return bm(pose_body=pose_body).Jtr

#             gt_j = decode_to_j(gt)
#             zv_j = decode_to_j(zv_p)
#             cv_j = decode_to_j(cv_p)

#             # 3. Calculate Error
#             for h_n, h_f in horizons.items():
#                 results["Zero-Vel"][h_n].append(compute_mpjpe(zv_j[:h_f], gt_j[:h_f]))
#                 results["Const-Vel"][h_n].append(compute_mpjpe(cv_j[:h_f], gt_j[:h_f]))

#     print("\n--- SUNDAY PROGRESS REPORT RESULTS ---")
#     for h in horizons:
#         avg_zv = sum(results["Zero-Vel"][h]) / len(results["Zero-Vel"][h])
#         avg_cv = sum(results["Const-Vel"][h]) / len(results["Const-Vel"][h])
#         print(f"Horizon {h}: Zero-Vel: {avg_zv:.2f}mm | Const-Vel: {avg_cv:.2f}mm")

# if __name__ == "__main__":
#     run_baseline_benchmarks()
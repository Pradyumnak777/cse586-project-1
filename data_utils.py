import torch
from torch.utils.data import Dataset
import os
import numpy as np
from vPoser_test import load_vposer, amass_poses_load, pose_encode, path_setup
#load data from the AMASS files, and feed this to the VAE

def make_latents(src_fps = 120, tgt_fps = None): #says in the description pdf that it was recorded at 120fps
    #load the VAE
    vposer_dir = 'VPoserModelFiles'
    amass_dir = 'AMASS_CMUsubset'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #get vposer model
    expr_dir = os.path.join(vposer_dir, 'vposer_v2_05/')
    vp, _ = load_vposer(expr_dir, device)
    
    #get all pose files in AMASS_CMUsubset
    pose_files = [f for f in os.listdir(amass_dir) if f.endswith('_poses.npz')]
    
    latents_dict = {}
    
    for pose_file in pose_files:
        file_path = os.path.join(amass_dir, pose_file)
        #load pose
        amass_body_pose = amass_poses_load(file_path, device) #shape: [T (i.e, 120), 63]. T is the number of frames
        if tgt_fps is not None:
            #note that tgt_fps shouldnt be below 0 or above 120..(which is the src_fps)
            #club the amass body poses that fall in the "frame umbrella"
            iter = max(1, int(round(src_fps / tgt_fps))) #keeping every 'iter'th frame
            '''
            if 30 fps, 120/30 = 4. keep every 4th frsame
            '''
            amass_body_pose = amass_body_pose[::iter] #shape: [30, 63]. new list
            
        
        #use vposer encoder
        latent_vectors = pose_encode(vp, amass_body_pose) #(T, D). num frames are not reduced..
        #store 
        key = os.path.splitext(pose_file)[0] 
        latents_dict[key] = latent_vectors.detach().cpu() #like {01_01_poses: latent}
    
    #save to disk
    # torch.save(latents_dict, os.path.join(path, 'vposer_latents.pt'))
    torch.save(latents_dict, 'vposer_latents.pt')

    
    # return latents_dict


class Data_VAE(Dataset):
    def __init__(self, latents_dict, window=1, context=1): #latents_dict is already tensors..
        self.train_data = [] #need to structure like - (curr_pose(s), last_curr_pose + 1(time)), for all the subjects/actions
        
        for k,v in latents_dict.items():
            #get frames for THIS video/pose seq
            num_frames = v.shape[0]
            
            for t in range(num_frames-(context+window)+1): #time
                curr_pose = v[t:t+context] #multiple frames for context
                next_pose = v[t+context+window-1]
                
                self.train_data.append((curr_pose, next_pose))
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        #for now, predicting consecutive frames. change later? change window param
        cur, next = self.train_data[idx]
        return cur, next #eg- ([f1, f2, f3], f4)


class Data_VAE_time(Dataset):
    def __init__(self, latents_dict, window_sec=1, context_sec=2, fps = 120): #latents_dict is already tensors..
        self.train_data = [] #need to structure like - (curr_pose(s), last_curr_pose + 1(time)), for all the subjects/actions
        self.context_frames = int(round(context_sec * fps))
        self.window_frames = int(round(window_sec * fps))
        
        self.window_frames = max(1, self.window_frames)
        
        for k,v in latents_dict.items():
            #get frames for THIS video/pose seq
            num_frames = v.shape[0]
            
            #making sure video is long neough
            if num_frames < (self.context_frames + self.window_frames):
                continue
            
            max_t = num_frames - (self.context_frames + self.window_frames) + 1
            
            for t in range(max_t): #time
                curr_pose = v[t:t+self.context_frames] #multiple frames for context
                target_idx = t + self.context_frames + self.window_frames - 1
                next_pose = v[target_idx]
                
                self.train_data.append((curr_pose, next_pose))
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        cur, next = self.train_data[idx]
        return cur, next #eg- ([contrxt_frames], target_frame after window_sec length of context)

        

if __name__ == "__main__":
    #save latents to disk
    make_latents()
    
    #import this dataset class in some other file after transformer arch is finished, to train.
    # latents_dict = torch.load('vposer_latents.pt')
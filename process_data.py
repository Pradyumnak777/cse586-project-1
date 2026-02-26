import torch
from torch.utils.data import Dataset
import os
import numpy as np
from vPoser_test import load_vposer, amass_poses_load, pose_encode, path_setup
#load data from the AMASS files, and feed this to the VAE

def make_latents():
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
        amass_body_pose = amass_poses_load(file_path, device)
        #use vposer encoder
        latent_vectors = pose_encode(vp, amass_body_pose)
        #store 
        key = os.path.splitext(pose_file)[0] 
        latents_dict[key] = latent_vectors.detach().cpu() #like {01_01_poses: latent}
    
    #save to disk
    # torch.save(latents_dict, os.path.join(path, 'vposer_latents.pt'))
    torch.save(latents_dict, 'vposer_latents.pt')

    
    # return latents_dict


class Data_VAE(Dataset):
    def __init__(self, latents_dict, window=1): #latents_dict is already tensors..
        self.train_data = [] #need to structure like - (curr_pose, curr_pose + 1(time)), for all the subjects/actions
        
        for k,v in latents_dict.items():
            #get frames for THIS video/pose seq
            num_frames = v.shape[0]
            
            for t in range(num_frames-window): #time
                curr_pose = v[t]
                next_pose = v[t+window]
                
                self.train_data.append((curr_pose, next_pose))
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        #for now, predicting consecutive frames. change later? change window param
        cur, next = self.train_data[idx]
        return cur, next
        
        

if __name__ == "__main__":
    #save latents to disk
    make_latents()
    
    #import this dataset class in some other file after transformer arch is finished, to train.
    # latents_dict = torch.load('vposer_latents.pt')
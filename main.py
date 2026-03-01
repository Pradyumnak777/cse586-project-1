import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_utils import Data_VAE

#this is where the model will hopefully be invoked and the training will actually happen.

latents_dict = torch.load('vposer_latents.pt')

k = list(latents_dict.keys()) #{name: latents} format. name is like '01_01_poses.npz'
train_poses, test_poses = train_test_split(k, test_size=0.2, random_state=420)

train_latents = {k: latents_dict[k] for k in train_poses}
test_latents  = {k: latents_dict[k] for k in test_poses}

train_dataset = Data_VAE(train_latents, window=1) #this dataset ism ade for continous (t, t+1) prediction
test_dataset  = Data_VAE(test_latents, window=1)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

'''
Note for @mayur: check if you can directly use these loaders for your stuff.
this will output- (current_frame, next_frame). is this useful for the velocity thing or whatever?
'''
print(len(train_dataset), len(test_dataset))
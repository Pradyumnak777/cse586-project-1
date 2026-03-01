import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_utils import Data_VAE

#this is where the model will hopefully be invoked and the training will actually happen.

latents_dict = torch.load('vposer_latents.pt', weights_only=True)

'''
CHANGE below for tweaking how dataloader works.
window = 1 means we want to predict the immediate next frame (ts is vannila, keep it for now)
context = 4 is the input to the transformer. means it'll take 4 consecutive input frame
'''
window = 1
context = 4

k = list(latents_dict.keys()) #{name: latents} format. name is like '01_01_poses.npz'
train_poses, test_poses = train_test_split(k, test_size=0.2, random_state=420)

train_latents = {k: latents_dict[k] for k in train_poses}
test_latents  = {k: latents_dict[k] for k in test_poses}

train_dataset = Data_VAE(train_latents, window=window, context=context) #this dataset ism ade for continous (t, t+1) prediction
test_dataset  = Data_VAE(test_latents, window=window, context=context)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

'''
Note for @mayur: check if you can directly use these loaders for your stuff.
this will output- ([curr frames based on context]], next_frame). is this useful for the velocity thing or whatever?
'''

print(len(train_dataset), len(test_dataset))


#some code ot check shapes

train_cur, train_next = next(iter(train_loader)) #should have another dim of len 4...cuz of context
print('train batch shapes:', train_cur.shape, train_next.shape)

test_cur, test_next = next(iter(test_loader)) #only 2 dims..
print('test batch shapes:', test_cur.shape, test_next.shape) 


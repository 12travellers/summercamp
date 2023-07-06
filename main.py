import model
import torch
import pymotionlib
from pymotionlib import BVHLoader
import numpy as np
import math
from tqdm import tqdm
import random

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

output_path = "./output"
data_path = "./walk1_subject5.bvh"
used_angles = 0
used_motions = 2
clip_size = 8
batch_size = 32
learning_rate = 1e-4
beta_VAE = 0.2
latent_size = 256

def build_data_set (data):
    dataset = []
    
    for i in range (data.shape[0] - clip_size):
        datapiece = data[i:i+clip_size, :]
        datapiece = datapiece.reshape ([1] + list(datapiece.shape))
        dataset.append (torch.tensor(datapiece))
    return torch.concat (dataset, dim = 0)


if __name__ == '__main__':

    encoder = model.VAE_encoder (motion_size, used_motions, 512, 256, latent_size, used_angles)
    decoder = model.VAE_decoder (motion_size, used_motions, latent_size, 512, 256, 3, used_angles)
    
    VAE = model.VAE(encoder, decoder)
    optimizer = torch.optim.Adam(VAE.parameters(), lr = learning_rate)
  
    try:
        checkpoint = torch.load (output_path + '/final_model.pth')
        VAE.load_state_dict (checkpoint['model'])
        epoch = checkpoint ['epoch']
        loss_history = checkpoint ['loss_history']
        print ("loading model from " + output_path + '/final_model.pth')
    except:
        print ("no training history found... please run train.py to build one...")
    
    bvh = BVHLoader.load (data_path)
    motions = bvh._joint_position
    motions = motions.reshape (bvh.num_frames, -1) / (math.pi * 2)
    motions = motions + 0.5
    
    motion_size = motions.shape [1]
    
    x = torch.zeros ([1, motion_size])
    anime_time = 100
    bvh = bvh.get_t_pose ()
    print( len(bvh._joint_rotation) )
    '''
    bvh.num_frames = 0
    
    while anime_time > 0:
        anime_time -= 1
        re_x, mu, sigma = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
        sample = torch.randn(1,latent_size)
        
        for i in range(len(bvh.name_list)):
            bvh._joint_rotation[:bvh.num_frames, index[i], :] = bvh_list[0]._joint_rotation[:, i, :]
      
        
        x = re_x
        
    BVHLoader.save(bvh, './infered.bvh')
        
        
'''
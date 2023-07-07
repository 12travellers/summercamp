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
beta_VAE = 0.006
latent_size = 256
area_width = 2

def build_data_set (data):
    dataset = []
    
    for i in range (data.shape[0] - clip_size):
        datapiece = data[i:i+clip_size, :]
        datapiece = datapiece.reshape ([1] + list(datapiece.shape))
        dataset.append (torch.tensor(datapiece))
    return torch.concat (dataset, dim = 0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("train model on device:" + str(device))
        
    bvh = BVHLoader.load (data_path)
    bvh = bvh.get_t_pose ()
    
    
    motions = bvh._joint_rotation
    motions = motions.reshape (bvh.num_frames, -1) / (math.pi * 2)
    motions = motions + 0.5
    
    translation = bvh._joint_translation
    translation = motions.reshape (bvh.num_frames, -1) / area_width
    translation = translation + 0.5
    
    # motion_size = real_motion_size + root_position_size
    motions = np.concatenate ([motions, translation [:, 0:3]], axis = 1)
    motion_size = motions.shape [1]
    

    encoder = model.VAE_encoder (motion_size, used_motions, 256, 256, latent_size, used_angles)
    decoder = model.VAE_decoder (motion_size, used_motions, latent_size, 256, 256, 3, used_angles)
    
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

    
    x = torch.zeros ([1, motion_size])
    anime_time = 100

    
    x = torch.tensor (motions [0:1, :] )
    
    
    
    while anime_time > 0:
        anime_time -= 1
        z = torch.randn(1, latent_size)
        
        re_x, moe_output = VAE.decoder (torch.concat ([x], dim = 1), z)
        trans = np.zeros(re_x[0, :-3].shape[0]//4*3)
        trans [0:3] = re_x [0, -3:].detach().numpy()
    
        
        bvh.append_trans_rotation ((trans.reshape([1, -1, 3]) - 0.5) * math.pi*2,\
            (re_x[0, :-3].reshape([1, -1, 4]).detach().numpy() - 0.5) * area_width)
        
        x = re_x
     
    #bvh.recompute_joint_global_info ()
    BVHLoader.save(bvh, './infered.bvh')
        
'''


python -m pymotionlib.editor



'''
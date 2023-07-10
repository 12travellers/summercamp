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
learning_rate = 4e-5
beta_VAE = 0.01
beta_para = 0.1
beta_moe = 0.2
h1 = 256
h2 = 128
moemoechu = 4
latent_size = 128
beta_trans = 10

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
    
    motions = bvh._joint_rotation
    motions = np.diff(motions, axis = 0)
    motions_min = np.min(motions,axis=(0,1,2))
    motions_max = np.max(motions,axis=(0,1,2))
    motions = motions.reshape (bvh.num_frames-1, -1) 
    
    translations = bvh._joint_translation
    translations = np.diff(translations, axis = 0)
    translations_min = np.min(translations,axis=(0,1,2))
    translations_max = np.max(translations,axis=(0,1,2))
    translations = translations.reshape (bvh.num_frames-1, -1)
    
    motions = np.concatenate ([motions, translations [:, 0:3]], axis = 1)
    motion_size = motions.shape [1]
    
    now_trans = np.mean(bvh._joint_translation, axis = 0)
    now_trans = now_trans.reshape([1]+list(now_trans.shape))
    now_rotat = np.mean(bvh._joint_rotation, axis = 0)
    now_rotat = now_rotat.reshape([1]+list(now_rotat.shape))
    
    bvh = bvh.get_t_pose ()
    
    
    # motion_size = real_motion_size + root_position_size
    
    

    encoder = model.VAE_encoder (motion_size, used_motions, h1, h2, latent_size, used_angles)
    decoder = model.VAE_decoder (motion_size, used_motions, latent_size, h1, h2, moemoechu, used_angles)
     
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
        exit (0)

    
    x = torch.zeros ([1, motion_size])
    anime_time = 1000

    
    x = torch.tensor (motions [0:1, :] )
    
    while anime_time > 0:
        anime_time -= 1
        z = torch.randn(1, latent_size)
        
        re_x, moe_output = VAE.decoder (torch.concat ([x], dim = 1), z)
        trans = np.zeros(re_x[0, :-3].shape[0]//4*3)
        trans [0:3] = re_x [0, -3:].detach().numpy()
        
        now_trans += trans.reshape([1, -1, 3]) * (translations_max - translations_min) + translations_min
        
        now_rotat += re_x[0, :-3].reshape([1, -1, 4]).detach().numpy() * (motions_max - motions_min) + motions_min
        bvh.append_trans_rotation (now_trans, now_rotat)
        
        x = re_x
     
    #bvh.recompute_joint_global_info ()
    BVHLoader.save(bvh, './infered.bvh')
        
'''


python -m pymotionlib.editor



'''
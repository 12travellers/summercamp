import model
import train
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
learning_rate = 4e-6
beta_VAE = 10
beta_grow_round = 10
beta_para = 0.1
beta_moe = 0.2
h1 = 256
h2 = 128
moemoechu = 4
latent_size = 128
beta_trans = 4
joint_num = 25
predicted_size = None
predicted_sizes = None
input_size = None
input_sizes = None


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
    
    try:
        checkpoint = torch.load (output_path + '/final_model.pth')
        
        input_size = checkpoint['input_size']
        input_sizes = checkpoint['input_sizes']
        predicted_size = checkpoint['predicted_size']
        predicted_sizes = checkpoint['predicted_sizes']
        
        translations_max = checkpoint["translations_max"]
        translations_min = checkpoint["translations_min"]
        motions_max = checkpoint["motions_max"]
        motions_min = checkpoint["motions_min"]
        
        
        
        encoder = model.VAE_encoder (input_size, used_motions, h1, h2, latent_size)
        decoder = model.VAE_decoder (input_size, used_motions, latent_size, predicted_size, h1, h2, moemoechu)
        
        VAE = model.VAE(encoder, decoder)
        optimizer = torch.optim.Adam(VAE.parameters(), lr = learning_rate)
        VAE.load_state_dict (checkpoint['model'])
        epoch = checkpoint ['epoch']
        loss_history = checkpoint ['loss_history']
        
        print ("loading model from " + output_path + '/final_model.pth')

        
        x = torch.zeros ([1, input_size])
        now_pos =
        now_ori = 
        anime_time = 1000
        
        joint_position, joint_orientation = [], []
        
        while anime_time > 0:
            anime_time -= 1
            z = torch.randn(1, latent_size)
            
            re_x, moe_output = VAE.decoder (torch.concat ([x], dim = 1), z)
            oripos = train.transform_as_predict(re_x)
            ori, pos = oripos[0], oripos[1]
            
            
            now_pos += pos.reshape([1, -1, 3]) * (translations_max - translations_min) + translations_min
            
            now_ori += ori.reshape([1, -1, 4]).detach().numpy() * (motions_max - motions_min) + motions_min
            bvh.append_trans_rotation (now_trans, now_rotat)
            
            x = re_x
        
        #bvh.recompute_joint_global_info ()
        
        joint_translation, joint_rotation = None, None
        joint_translation, joint_rotation = \
            bvh.compute_joint_local_info (joint_position, joint_orientation, joint_translation, joint_rotation)
        BVHLoader.save(bvh, './infered.bvh')
        
        
        
    except:
        print ("no training history found... please run train.py to build one...")
        exit (0)
        
'''


python -m pymotionlib.editor



'''
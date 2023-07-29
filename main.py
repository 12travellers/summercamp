import model
import train
import torch
import pymotionlib
from pymotionlib import BVHLoader, MotionData
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import math
from tqdm import tqdm
import random

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from train import *


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("train model on device:" + str(device))
    
    checkpoint = torch.load (output_path + '/final_model.pth')
    
    input_size = checkpoint['input_size']
    input_sizes = checkpoint['input_sizes']
    predicted_size = checkpoint['predicted_size']
    predicted_sizes = checkpoint['predicted_sizes']
    
    train.inputs_std = checkpoint["inputs_std"]
    train.inputs_avg = checkpoint["inputs_avg"]
    train.inputs_avg_gpu, train.inputs_std_gpu =\
        torch.tensor(train.inputs_avg).to(torch.float32).detach().to(device), \
        torch.tensor(train.inputs_std).to(torch.float32).detach().to(device)

    
    
    bvh = BVHLoader.load (data_path).sub_sequence(514,516)
    bvh.recompute_joint_global_info ()
    jt, jr = bvh.joint_translation[-1], bvh._joint_rotation[-1]
    
    if True:
        for i in range(10):
            root_ori_b, root_ori = bvh._joint_orientation[-2,i],  bvh._joint_orientation[-1,i]
            angular_velocity = bvh.compute_angular_velocity (False)[-1, i]
            print(i,":",train.calc_root_ori(root_ori_b, angular_velocity, bvh), root_ori)
    
    
    
    motions, translations, root_info = train.transform_bvh(bvh)
    inputs = np.concatenate ([motions, translations], axis = 1)
    inputs = train.move_input_to01(inputs)
    
    
    encoder = model.VAE_encoder (input_size, used_motions, h1, h2, latent_size)
    decoder = model.VAE_decoder (input_size, used_motions, latent_size, input_size, h2, h1, moemoechu)
    
    VAE = model.VAE(encoder, decoder)
    optimizer = torch.optim.Adam(VAE.parameters(), lr = learning_rate)
    VAE.load_state_dict (checkpoint['model'])
    epoch = checkpoint ['epoch']
    loss_history = checkpoint ['loss_history']
    
    print ("loading model from " + output_path + '/final_model.pth')

    
    x = torch.from_numpy (inputs[-1])
    x = x.reshape([1] + list(x.shape))
    anime_time = 300
    root_ori, root_pos = root_info[-1][0], root_info[-1][1]
    
    
    while anime_time > 0:
        anime_time -= 1
        z = torch.randn([1, latent_size])
        
        re_x, moe_output = VAE.decoder (x, z)
        
        #test basic function
        #re_x = torch.from_numpy(inputs[514+1000-anime_time])
        #re_x = re_x.reshape([1] + list(re_x.shape))
    
        x = x.numpy()
        re_x = re_x.detach().numpy()
        
        x,= train.move_input_from01 (x),
          #      ,train.move_input_from01(re_x, motions_max, motions_min, translations_max, translations_min, input_sizes[0])
                
   
        root_ori, root_pos = train.transform_root_from_input(re_x[0], root_ori, root_pos, bvh)
        
        jt_b, jr_b = jt, jr
        jt, jr = train.compute_motion_info(re_x[0], root_pos, root_ori, jt, jr, bvh)
        
        bvh.append_trans_rotation (np.asarray([jt]), np.asarray([jr]))
        #different methods
        #bd = bvh.sub_sequence(0,-1)
        #bd.append_trans_rotation (np.asarray([jt_b, jt]), np.asarray([jr_b, jr]))
        motions, translations, root_info = train.transform_bvh(bvh, bvh.num_frames-1)
        inputs = np.concatenate ([motions, translations], axis = 1)
        inputs = move_input_to01(inputs)
        x = torch.from_numpy (inputs[-1])
        x = x.reshape([1] + list(x.shape))
        
        #x=train.move_input_to01 (re_x, motions_max, motions_min, translations_max, translations_min, input_sizes[0])
        x=torch.tensor(x).detach()
        
    #
    
    optimizer.zero_grad()
    #bvh.recompute_joint_global_info ()
    BVHLoader.save(bvh.sub_sequence(0, bvh.num_frames), './infered.bvh')
    
    os.system("python -m pymotionlib.editor")
    os.system("tensorboard --logdir=./runs")
    

        
'''


python -m pymotionlib.editor

tensorboard --logdir=.  
http://localhost:6006/

'''
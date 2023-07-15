import model
import train
import torch
import pymotionlib
from pymotionlib import BVHLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
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
    
    checkpoint = torch.load (output_path + '/final_model.pth')
    
    input_size = checkpoint['input_size']
    input_sizes = checkpoint['input_sizes']
    predicted_size = checkpoint['predicted_size']
    predicted_sizes = checkpoint['predicted_sizes']
    
    translations_max = checkpoint["translations_max"]
    translations_min = checkpoint["translations_min"]
    motions_max = checkpoint["motions_max"]
    motions_min = checkpoint["motions_min"]

    bvh = BVHLoader.load (data_path).sub_sequence(114, 116)
    motions, translations, root_info = train.transform_bvh(bvh)
    print(motions.shape, translations.shape)
    motions = (motions - motions_min) / (motions_max - motions_min)
    translations = (translations - translations_min) / (translations_max - translations_min)
    inputs = np.concatenate ([motions, translations], axis = 1)
    
    
    
    encoder = model.VAE_encoder (input_size, used_motions, h1, h2, latent_size)
    decoder = model.VAE_decoder (input_size, used_motions, latent_size, predicted_size, h1, h2, moemoechu)
    
    VAE = model.VAE(encoder, decoder)
    optimizer = torch.optim.Adam(VAE.parameters(), lr = learning_rate)
    VAE.load_state_dict (checkpoint['model'])
    epoch = checkpoint ['epoch']
    loss_history = checkpoint ['loss_history']
    
    print ("loading model from " + output_path + '/final_model.pth')

    
    x = torch.from_numpy (inputs[-1])
    x = x.reshape([1] + list(x.shape))
    anime_time = 1000
    root_ori, root_pos = root_info[1][0], root_info[1][1]
    
    joint_translations, joint_rotations = [], []
    while anime_time > 0:
        anime_time -= 1
        z = torch.randn([1, latent_size])
        
        re_x, moe_output = \
            VAE.decoder (x, z)
    

        x = x.numpy()
        re_x = re_x.detach().numpy()
        print(root_ori.shape, root_pos.shape)
        x[0] = train.transform_as_input (x[0], re_x[0], root_ori, root_pos, bvh)
        x=torch.tensor(x)
        
        
        root_ori, root_pos = train.transform_root(re_x[0], root_ori, root_pos)
        
        joint_translation, joint_rotation = train.compute_motion_info(re_x[0], root_ori, root_pos, bvh)
        
        joint_translations.append(joint_translation)
        joint_rotations.append(joint_rotation)
        
    
    #bvh.recompute_joint_global_info ()
    
    bvh.append_trans_rotation (joint_translations, joint_rotations)
    
    BVHLoader.save(bvh.sub_sequence(10, bvh.num_frames), './infered.bvh')
    
    
    

        
'''


python -m pymotionlib.editor

tensorboard --logdir=.  
http://localhost:6006/

'''
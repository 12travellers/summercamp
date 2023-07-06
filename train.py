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
area_width = 256

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
    motions = motions.reshape (bvh.num_frames, -1) / (math.pi * 2)
    motions = motions + 0.5
    
    translation = bvh._joint_translation
    translation = motions.reshape (bvh.num_frames, -1) / area_width
    translation = translation + 0.5
    
    # motion_size = real_motion_size + root_position_size
    motions = np.concatenate ([motions, translation [:, 0:3]], axis = 1)
    motion_size = motions.shape [1]
    
    print("read " + str(motions.shape) + "motions from " + data_path)
    
    train_motions, test_motions = train_test_split(motions, test_size = 0.1)
    
    encoder = model.VAE_encoder (motion_size, used_motions, 256, 256, latent_size, used_angles)
    decoder = model.VAE_decoder (motion_size, used_motions, latent_size, 256, 256, 3, used_angles)
    
    VAE = model.VAE(encoder, decoder)
    optimizer = torch.optim.Adam(VAE.parameters(), lr = learning_rate)
    
    iteration = 12
    epoch = 0
    p0_iteration, p1_iteration = 4, 2
    loss_history = {'train':[], 'test':[]}
    
    
    try:
        checkpoint = torch.load (output_path + '/final_model.pth')
        VAE.load_state_dict (checkpoint['model'])
        epoch = checkpoint ['epoch']
        loss_history = checkpoint ['loss_history']
        print ("loading model from " + output_path + '/final_model.pth')
    except:
        print ("no training history found... rebuild another model...")
        
    
    
    train_loader = torch.utils.data.DataLoader(\
        dataset = build_data_set (train_motions),\
        batch_size = batch_size,\
        shuffle = True)
    test_loader = torch.utils.data.DataLoader(\
        dataset = build_data_set (test_motions),\
        batch_size = batch_size,\
        shuffle = True)
    
    loss_MSE = torch.nn.MSELoss(reduction = 'sum')
    loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)


    while (epoch < iteration):
        teacher_p = 0
        if (epoch < p0_iteration):
            teacher_p = (p1_iteration - epoch) / (p1_iteration -p0_iteration)
        elif(epoch < p1_iteration):
            teacher_p = 1
        epoch += 1 
        
        t = tqdm(train_loader, desc = f'[train]epoch:{epoch}')
        train_loss, train_nsample = 0, 0
         
        for motions in train_loader:
            x = motions [:, 0, :]
            for i in range (1, clip_size):
                re_x, mu, sigma = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
                
                loss_re = loss_MSE(re_x.to (torch.float32), motions [:, i, :].to (torch.float32))
                loss_norm = loss_KLD(mu, sigma)
                loss = loss_re + beta_VAE * loss_norm
                
                if (random.random() < teacher_p):
                    x = motions [:, i, :]
                else:
                    x = re_x
                
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_nsample += batch_size * (clip_size - 1)
            t.set_postfix({'loss':train_loss/train_nsample})
            
        loss_history['train'].append(train_loss/train_nsample)

        
        
        if (epoch % 3 == 0 and epoch > p0_iteration):
            
            test_loss, test_nsample = 0, 0
            
            for motions in test_loader:
                x = motions [:, 0, :]
                for i in range (1, clip_size):
                    re_x, mu, sigma = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
                    
                    loss_re = loss_MSE(re_x.to (torch.float32), motions [:, i, :].to (torch.float32))
                    loss_norm = loss_KLD(mu, sigma)
                    loss = loss_re + beta_VAE * loss_norm
                    
                    if (random.random() < teacher_p):
                        x = motions [:, i, :]
                    else:
                        x = re_x
                
                test_loss += loss.item()
                test_nsample += batch_size* (clip_size - 1)
                t.set_postfix ({'loss':test_loss/test_nsample})  
            print ("iteration %d/%d, test_loss: %f", epoch, iteration, test_loss/test_nsample)
        
                
        state = {'model': VAE.state_dict(),\
                 'epoch': epoch,\
                 'loss_history': loss_history}
        torch.save(state, output_path+'/final_model.pth')
        print ("iteration %d/%d, train_loss: %f", epoch, iteration, train_loss/train_nsample)
    
    
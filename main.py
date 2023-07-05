import model
import torch
import pymotionlib
from pymotionlib import BVHLoader
import numpy as np
import tqdm
import random

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

output_path = "./output"
data_path = "./walk1_subject5.bvh"
used_angles = 0
used_motions = 1
clip_size = 8
batch_size = 32
learning_rate = 1e-4
beta_VAE = 0.2


def build_data_set (data):
    dataset = torch.zeros ([0, clip_size, data.shape[1]])
    for i in range (data.shape[0] - clip_size):
        dataset.append (data[i:i+clip_size, :])
    return dataset


if __name__ == '__main__':
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    bvh = BVHLoader.load (data_path)
    motions = bvh._joint_position
    motions = motions.reshape (bvh.num_frames, -1)
    
    motion_size = motions.shape [1]
    
    print("read " + str(motions.shape) + "motions from " + data_path)
    
    train_motions, test_motions = train_test_split(motions, test_size = 0.1)
    
    encoder = model.VAE_encoder (motion_size, used_motions, 512, 256, 256, used_angles)
    decoder = model.VAE_decoder (motion_size, used_motions, latent_size, 512, 256, 3, used_angles)
    
    VAE = model.VAE(encoder, decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    iteration = 100
    p0_iteration, p1_iteration = 40, 20
    
    checkpoint = torch.load (output_path+'/final_model.pth')
    VAE.load_state_dict (checkpoint['model'])
    epoch = checkpoint (['epoch'])
    loss_history = checkpoint (['loss_history'])
    
    train_loader = torch.utils.data.DataLoader(\
        dataset = build_data_set (train_motions),\
        batch_size = batch_size,\
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(\
        dataset = build_data_set (test_motions),\
        batch_size = batch_size,\
        shuffle=True)
    loss_BCE = torch.nn.BCELoss(reduction = 'sum')
    loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)


    while (epoch < iteration):
        teacher_p = 1
        if (epoch < p0_iteration):
            teacher_p = (p1_iteration - epoch) / (p1_iteration -p0_iteration)
        else:
            teacher_p = 0 
        epoch += 1 
        
        t = tqdm(train_loader, desc = f'[train]epoch:{epoch}')
        train_loss, train_nsample = 0, 0
         
        for step, motions in train_loader:
            x = [motions [:, 0, :]
            for i in range (1, clip_size):
                re_x, mu, sigma = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
                
                loss_re = loss_BCE(re_x, motions [:, i, :])
                loss_norm = loss_KLD(mu, sigma)
                loss = loss_re + beta_VAE * loss_norm
                
                for j in range(0, batch_size):
                    if (random.random() < teacher_p):
                        re_x [:, j, :] = motions [:, i, :]
                x = re_x
                
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_nsample += batch_size* (clip_size - 1)
            t.set_postfix({'loss':train_loss/train_nsample})
            
        loss_history['train'].append(train_loss/train_nsample)
        
        state = {'model': VAE.state_dict(),\
                 'epoch': epoch,\
                 'loss_histort': loss_history}
        torch.save(state, output_path+'/final_model.pth')
        print ("iteration %d/%d, train_loss: %f", epoch, iteration, train_loss/train_nsample)
        
        '''
        if (epoch % 10 == 0 and epoch > p0_iteration):
            
            test_loss, test_nsample = 0, 0
            
            for step, motions in test_loader:
                x = motions [:, 0, :]
                for i in range (1, clip_size):
                    re_x, mu, sigma = VAE.decoder (x, z)
                    
                    loss_re = loss_BCE(re_x, motions [:, i, :]) # 重构与原始数据的差距(也可使用loss_MSE)
                    loss_norm = loss_KLD(mu, sigma) # 正态分布(mu,sigma)与正态分布(0,1)的差距
                    loss = loss_re + beta_VAE * loss_norm
                    
                    x = re_x
                    
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                test_loss += loss.item()
                test_nsample += batch_size* (clip_size - 1)
                t.set_postfix ({'loss':test_loss/test_nsample})  
            print ("iteration %d/%d, test_loss: %f", epoch, iteration, test_loss/test_nsample)
        '''
    
    
    
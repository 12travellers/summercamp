import model
import torch
import pymotionlib
from pymotionlib import BVHLoader
import numpy as np
import math
from tqdm import tqdm
import random
import sys

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
beta_para = 0.1
beta_moe = 0.6
h1 = 256
h2 = 128
moemoechu = 4
latent_size = 128
beta_trans = 4

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

    make_new = False
    if (len(sys.argv) > 0):
        print(str(sys.argv))
        if (str(sys.argv)[1] == '-n'):
            make_new =  True
    
    bvh = BVHLoader.load (data_path)
    
    motions = bvh._joint_rotation
    motions_min = np.min(motions)
    motions_max = np.max(motions)
    motions = (motions - motions_min) / (motions_max - motions_min)
    motions = motions.reshape (bvh.num_frames, -1) 
    
    translations = bvh._joint_translation
    translations_min = np.min(translations)
    translations_max = np.max(translations)
    translations = (translations - translations_min) / (translations_max - translations_min)
    translations = translations.reshape (bvh.num_frames, -1)
    
    
    
    # motion_size = real_motion_size + root_position_size
    motions = np.concatenate ([motions, translations [:, 0:3]], axis = 1)
    motion_size = motions.shape [1]
    
    print("read " + str(motions.shape) + "motions from " + data_path)
    
    train_motions, test_motions = train_test_split(motions, test_size = 0.1)
    
    encoder = model.VAE_encoder (motion_size, used_motions, h1, h2, latent_size, used_angles)
    decoder = model.VAE_decoder (motion_size, used_motions, latent_size, h1, h2, moemoechu, used_angles)
    
    VAE = model.VAE(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr = learning_rate)
    
    iteration = 100
    epoch = 0
    p0_iteration, p1_iteration = 40, 20
    loss_history = {'train':[], 'test':[]}
    
    
    try:
        assert (False == make_new)
        checkpoint = torch.load (output_path + '/final_model.pth')
        VAE.load_state_dict (checkpoint['model'])
        epoch = checkpoint ['epoch']
        loss_history = checkpoint ['loss_history']
        optimizer = checkpoint ['optimizer']
        print ("loading model from " + output_path + '/final_model.pth')
    except:
        print ("no training history found... rebuild another model...")
        
    
    
    train_loader = torch.utils.data.DataLoader(\
        dataset = build_data_set (train_motions).to(device),\
        batch_size = batch_size,\
        shuffle = True)
    test_loader = torch.utils.data.DataLoader(\
        dataset = build_data_set (test_motions).to(device),\
        batch_size = batch_size,\
        shuffle = False)
    
    loss_MSE = torch.nn.MSELoss(reduction = 'sum')
    loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)

    while (epoch < iteration):
        teacher_p = 0
        if (epoch < p0_iteration):
            teacher_p = (p1_iteration - epoch) / (p0_iteration - p1_iteration)
        elif(epoch < p1_iteration):
            teacher_p = 1
        epoch += 1 
        
        t = tqdm (train_loader, desc = f'[train]epoch:{epoch}')
        train_loss, train_nsample = 0, 0
        
        for motions in train_loader:
            x = motions [:, 0, :]
            for i in range (1, clip_size):
                re_x, mu, sigma, moe_output = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
                    
                loss_re = loss_MSE(re_x[:, :-3], motions [:, i, :-3].to(torch.float32)) +\
                    loss_MSE(re_x[:, -3:], motions [:, i, -3:].to(torch.float32)) * beta_trans
                loss_moe = 0
                
                moemoe, moemoepara = moe_output
                for j in range(moemoechu):
                    re = torch.mul(moemoepara[:, :, j:j+1], moemoe[j, : :])
                    gt = torch.mul(moemoepara[:, :, j:j+1], motions [:, i, :]).to(torch.float32)
                    loss_moe += loss_MSE(re[:, :-3], gt [:, :-3]) +\
                    loss_MSE(re[:, -3:], gt[:, -3:]) * beta_trans

                loss_para = torch.sum (torch.mul (moemoepara, moemoepara), dim = (0, 1, 2))
        
                
                loss_norm = loss_KLD(mu, sigma)
                loss = loss_re + beta_VAE * loss_norm + beta_moe * loss_moe + beta_para * loss_para
            #    print(loss_re, loss_norm, loss_moe, moemoepara[:, :, j])
                if (random.random() < teacher_p):
                    x = motions [:, i, :]
                else:
                    x = re_x
                
                if (train_nsample == 0):
                    print (loss_re.item(), beta_VAE*loss_norm.item(), beta_moe*loss_moe.item(), beta_para*loss_para.item())
                
            
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_nsample += batch_size * (clip_size - 1)
            t.set_postfix({'loss':train_loss/train_nsample})
            
        loss_history['train'].append(train_loss/train_nsample)

        
        
        if (epoch % 500000 == 0 and epoch > p0_iteration):
            
            test_loss, test_nsample = 0, 0
            
            for motions in test_loader:
                x = motions [:, 0, :]
                for i in range (1, clip_size):
                    re_x, mu, sigma, moe_output = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
                    
                    loss_re = loss_MSE(re_x, motions [:, i, :].to(torch.float32))
                    loss_moe, loss_para = 0, 0
                    
                    moemoe, moemoepara = moe_output
                    for j in range(moemoechu):
                        loss_moe += loss_MSE(torch.mul(moemoepara[:, :, j:j+1], moemoe[j, : :]), \
                            torch.mul(moemoepara[:, :, j:j+1], motions [:, i, :].to(torch.float32)))
                    loss_norm = loss_KLD(mu, sigma)
                    loss = loss_re + beta_VAE * loss_norm + beta_moe * loss_moe + beta_para * loss_para
                
                    
                    x = re_x
                
                test_loss += loss.item()
                test_nsample += batch_size* (clip_size - 1)
                t.set_postfix ({'loss':test_loss/test_nsample})  
            print ("iteration %d/%d, test_loss: %f", epoch, iteration, test_loss/test_nsample)
        
                    
        state = {'model': VAE.state_dict(),\
                    'epoch': epoch,\
                    'loss_history': loss_history,\
                    'optimizer': optimizer}
        torch.save(state, output_path+'/final_model.pth')
        print ("iteration %d/%d, train_loss: %f", epoch, iteration, train_loss/train_nsample)
        
    
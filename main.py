import model
import torch
import pymotionlib
from pymotionlib import BVHLoader
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split







if __name__ == '__main__':
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = "./output"
    data_path = "./walk1_subject5.bvh"
    used_angles = 0
    used_motions = 1
    clip_size = 16
    
    bvh = BVHLoader.load (data_path)
    motions = bvh._joint_position
    motions = motions.reshape (bvh.num_frames, -1)
    
    motion_size = motions.shape [1]
    
    print("read " + str(motions.shape) + "motions from " + data_path)
    
    train_motions, test_motions = train_test_split(motions, test_size = 0.1)
    
    encoder = model.VAE_encoder (motion_size, used_motions, 512, 256, 256, used_angles)
    decoder = model.VAE_decoder (motion_size, used_motions, latent_size, 512, 256, 3, used_angles)
    
    VAE = model.VAE(encoder, decoder)
    
    iteration = 100
    p0_iteration, p1_iteration = 40, 20
    
    checkpoint = torch.load (output_path+'/final_model.pth')
    VAE.load_state_dict (checkpoint['model'])
    epoch = checkpoint (['epoch'])

    while (epoch < iteration):
        teacher_p = 1
        if (epoch < p0_iteration):
            teacher_p = (p1_iteration - epoch) / (p1_iteration -p0_iteration)
        else 
            teacher_p = 0 
        epoch += 1 
        
        
        
        imgs = imgs.to(device).view(bs,input_size) #imgs:(bs,28*28)
        #模型运算     
        re_imgs, mu, sigma = model(imgs)
        #计算损失
        loss_re = loss_BCE(re_imgs, imgs) # 重构与原始数据的差距(也可使用loss_MSE)
        loss_norm = loss_KLD(mu, sigma) # 正态分布(mu,sigma)与正态分布(0,1)的差距
        loss = loss_re + loss_norm
        #反向传播、参数优化，重置
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #计算平均损失，设置进度条
        train_loss += loss.item()
        train_nsample += bs
        t.set_postfix({'loss':train_loss/train_nsample})
           
        
        
        
        
        state = {'model': VAE.state_dict(), \
                 'epoch': epoch}
        torch.save(state, output_path+'/final_model.pth')
        print ("iteration %d/%d, train_loss: %f", %epoch, %iteration, %total_loss)
        if (epoch % 10 == 0):
            print ("iteration %d/%d, train_loss: %f", %epoch, %iteration, %test_loss)
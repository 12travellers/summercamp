import model
import torch
import pymotionlib
import shutil
from pymotionlib import BVHLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from tqdm import tqdm
import random
import sys

from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

output_path = "./output"
data_path = "./walk1_subject5.bvh"
used_angles = 0
used_motions = 2
clip_size = 8
batch_size = 32
learning_rate = 4e-6
beta_VAE = 1
beta_grow_round = 10
beta_para = 0
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
num_frames = None

class motion_data_set(torch.utils.data.Dataset):
    def __init__(self, data, root_info):
        self.dataset, self.root_ori, self.root_pos = [], [], []
        for i in range (data.shape[0] - clip_size):
            datapiece = data[i:i+clip_size, :]
            datapiece = datapiece
            self.dataset.append (torch.tensor(datapiece).to(device))
            self.root_ori.append (root_info[i+1][0])
            self.root_pos.append (root_info[i+1][1])
    
    def __getitem__(self, index):
        return self.dataset[index], self.root_ori[index], self.root_pos[index]
    
    def __len__(self):
        return len(self.dataset)

def build_data_set (data, root_info):
    dataset = []
    
    for i in range (data.shape[0] - clip_size):
        datapiece = data[i:i+clip_size, :]
        datapiece = datapiece.reshape ([1] + list(datapiece.shape))
        dataset.append (torch.tensor(datapiece))
    return torch.concat (dataset, dim = 0)


def move_to_01 (data):
    _min, _max = np.min(data), np.max(data)
    return (data - _min) / (_max - _min), _min, _max

def transform_bvh (bvh):
    global predicted_size, predicted_sizes, input_size, input_sizes, num_frames
    #assume root is at index 0
    
    bvh = bvh.recompute_joint_global_info()
    
    linear_velocity = bvh.compute_linear_velocity (False)
    angular_velocity = bvh.compute_angular_velocity (False)
    position, orientation = bvh._joint_position, bvh._joint_orientation
    
    motions, translations = [], []
    num_frames = bvh.num_frames
    for i in range(1, bvh.num_frames):
        motion, translation = [], []
        
        root_ori = R(orientation[i, 0, :], normalize=False, copy=False)
        root_ori = root_ori.inv()
        
        for j in range(1, len(bvh._skeleton_joints)):
            translation.append (root_ori.apply(position[i, j] - position[i, 0]))
            motion.append ( (root_ori * R(orientation[i, j])).as_quat())
        
        for j in range(0, len(bvh._skeleton_joints)):
            if (j == 0):
                motion.append (linear_velocity[i, j])
                translation.append (angular_velocity[i, j])
            else:
                motion.append (root_ori.apply(linear_velocity[i, j]))
                translation.append ((root_ori * R.from_rotvec(angular_velocity[i, j])).as_quat())
            if (j == 0 and predicted_size == None):
                predicted_sizes = [\
                    np.concatenate (motion, axis = -1).shape[-1],\
                    np.concatenate (translation, axis = -1).shape[-1]]
                predicted_size = predicted_sizes [0] + predicted_sizes [1] 
                    
        
        motion = np.concatenate (motion, axis = -1)
        translation = np.concatenate (translation, axis = -1)
        if (input_size == None):
            input_sizes = [motion.shape[-1], translation.shape[-1]]
            input_size = motion.shape[-1] + translation.shape[-1]
            
        motions.append (motion.reshape([1] + list(motion.shape)))
        translations.append (translation.reshape([1] + list(translation.shape)))
    
    motions = np.concatenate (motions, axis = 0)
    translations = np.concatenate (translations, axis = 0)
    
    root_info = [(orientation[i, 0, :], position[i, 0, :]) for i in range(0, num_frames)]
    
    return motions, translations, root_info

def transform_as_predict (o):
    print(o.shape, predicted_sizes[1])
    return [o[:, :predicted_sizes[0]],\
        o[:, input_sizes[0]:input_sizes[0] + predicted_sizes[1]]]

def compute_motion_info (x, root_ori, root_pos):
    joint_position, joint_orientation = [root_pos.detach().numpy()], [root_ori.detach().numpy()]
    bs = predicted_sizes[0]
    
    root_ori = R(root_ori.detach().numpy())
    
    
    for i in range(0, joint_num-1):
        joint_position.append(root_ori.apply(x[bs+i*3:bs+i*3+3]) + root_pos.detach().numpy())
        joint_orientation.append((root_ori * R(x[i*4:i*4+4])).as_quat())
    
    print(joint_position[0].shape)
    joint_translation, joint_rotation = None, None
    joint_translation, joint_rotation =\
        bvh.compute_joint_local_info (joint_position, joint_orientation, joint_translation, joint_rotation)
    return joint_translation, joint_rotation

def transform_root (re_x, root_ori_b, root_pos_b):
    ori, pos = re_x [:predicted_sizes[0]], re_x[predicted_sizes[0]:]
    root_ori = root_ori_b + ori[-4:]
    root_pos = root_pos_b + pos[-3:]
    return root_ori, root_pos
def transform_as_input (x, re_x, root_ori_b, root_pos_b):
    
    ori, pos = re_x [:predicted_sizes[0]], re_x[predicted_sizes[0]:]
    root_ori = root_ori_b + ori[-4:]
    root_pos = root_pos_b + pos[-3:]
    
    jt_b, jr_b = compute_motion_info(x, root_ori_b, root_pos_b)
    jt, jr = compute_motion_info(re_x, root_ori, root_pos)
    
    root_ori = R(root_ori, normalize=False, copy=False)
    root_ori = root_ori.inv()
    extra_t, extra_r = []
    for j in range(1, num_frames):
        extra_r.append(root_ori.apply(jr[j] - jr_b[j]))
        extra_t.append(root_ori.apply(jt[j] - jt_b[j]))
    
    return torch.concat ([ori[:-4], root_ori] + extra_r +\
                         [pos[:-3], root_pos] + extra_t, dim = -1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("train model on device:" + str(device))
    
    make_new = False
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'new'):
            make_new =  True
    
    
    
    bvh = BVHLoader.load (data_path)
    print("read " + str(bvh.num_frames) + " motions from " + data_path)
    
    motions, translations, root_info = transform_bvh(bvh)
    
    
    motions, motions_min, motions_max = move_to_01 (motions)
    translations, translations_min, translations_max = move_to_01 (translations)
    
    inputs = np.concatenate ([motions, translations], axis = 1)
    assert (input_size == inputs.shape [1])
    
    
    train_motions, test_motions = train_test_split(inputs, test_size = 0.01)
    
    encoder = model.VAE_encoder (input_size, used_motions, h1, h2, latent_size)
    decoder = model.VAE_decoder (input_size, used_motions, latent_size, predicted_size, h1, h2, moemoechu)
    
    VAE = model.VAE(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr = learning_rate)
    iteration = 60
    epoch = 0
    p0_iteration, p1_iteration = 50, 30
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
        try:
            shutil.rmtree("runs/vae")
        except:
            None
        print ("no training history found... rebuild another model...")
        
    writer = SummaryWriter(log_dir='runs/vae')
    
    
    train_loader = torch.utils.data.DataLoader(\
        dataset = motion_data_set (train_motions, root_info),\
        batch_size = batch_size,\
        shuffle = True)
    
    loss_MSE = torch.nn.MSELoss(reduction = 'sum')
    loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)




    while (epoch < iteration):
        teacher_p = 0
        if (epoch < p0_iteration):
            teacher_p = (p1_iteration - epoch) / (p0_iteration - p1_iteration)
        elif(epoch < p1_iteration):
            teacher_p = 1
        epoch += 1 
        
        
        ###
        teacher_p = 0
        
        t = tqdm (train_loader, desc = f'[train]epoch:{epoch}')
        train_loss, train_nsample = 0, 0
        tot_loss_re , tot_loss_norm, tot_loss_moe, tot_loss_para = 0,0,0,0
        
        beta_VAE2 = beta_VAE
        if (epoch < beta_grow_round):
            beta_VAE2 = beta_VAE / beta_grow_round * beta_VAE2
        for motions, root_ori, root_pos in train_loader:
            x = motions [:, 0, :]
            for i in range (1, clip_size):
                re_x, mu, sigma, moe_output = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
                
                gt = torch.concat(transform_as_predict(motions [:, i, :]), axis=1).to(torch.float32)
                
                loss_re = loss_MSE(re_x, gt)
                loss_moe = 0
                
                moemoe, moemoepara = moe_output
                for j in range(moemoechu):
                    re = torch.mul(moemoepara[:, :, j:j+1], moemoe[j, : :])
                    gtp = torch.mul(moemoepara[:, :, j:j+1], gt).to(torch.float32)
                    loss_moe += loss_MSE(re, gtp) * beta_trans

                loss_para = torch.sum (torch.mul (moemoepara, moemoepara), dim = (0, 1, 2))
                loss_norm = loss_KLD(mu, sigma)
                loss = loss_re + beta_VAE2 * loss_norm + beta_moe * loss_moe + beta_para * loss_para
           
                tot_loss_re += loss_re.item()
                tot_loss_norm += beta_VAE * loss_norm.item()
                tot_loss_moe += beta_moe * loss_moe.item()
                tot_loss_para += beta_para * loss_para.item()
                train_nsample += (clip_size-1) * batch_size
                train_loss += loss.item()
            
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
           
                x, re_x = x.detach(), re_x.detach()
                if (random.random() < teacher_p):
                    x = motions [:, i, :]
                else:
                    for j in range(0, batch_size):
                        x[j] =\
                            transform_as_input (x[j].cpu(), re_x[j].cpu(),\
                                root_ori[j], root_pos[j]).device()
                    root_ori[j], root_pos[j] = transform_root(re_x, root_ori[j], root_pos[j])
                    

                
            train_loss += loss.item()
            train_nsample += batch_size * (clip_size - 1)
            t.set_postfix({'loss':train_loss/train_nsample})
            
        loss_history['train'].append(train_loss/train_nsample)

        writer.add_scalar(tag="loss_re",
            scalar_value=tot_loss_re/train_nsample,
            global_step=epoch)
        writer.add_scalar(tag="loss_norm",
            scalar_value=tot_loss_norm/train_nsample,
            global_step=epoch)
        writer.add_scalar(tag="loss_moe",
            scalar_value=tot_loss_moe/train_nsample,
            global_step=epoch)
        writer.add_scalar(tag="loss_para",
            scalar_value=tot_loss_para/train_nsample,
            global_step=epoch)
                  
        state = {'model': VAE.state_dict(),\
                    'epoch': epoch,\
                    'loss_history': loss_history,\
                    'optimizer': optimizer,\
                    'input_size': input_size,\
                    'input_sizes': input_sizes,\
                    'predicted_size': predicted_size,\
                    'predicted_sizes': predicted_sizes,\
                    'motions_min': motions_min,\
                    'motions_max': motions_max,\
                    'translations_min': translations_min,\
                    'translations_max': translations_max,\
                    }
        torch.save(state, output_path+'/final_model.pth')
        print ("iteration %d/%d, train_loss: %f", epoch, iteration, train_loss/train_nsample)
        
    
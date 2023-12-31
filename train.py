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

from pymotionlib.Utils import quat_product

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

output_path = "./output"
data_path = "./walk1_subject5.bvh"
used_angles = 0
used_motions = 2
clip_size = 16
batch_size = 64
learning_rate = 1e-4
beta_VAE = 0.02
beta_grow_round = 1
beta_para = 0
beta_moe = 0.4
h1 = 256
h2 = 128
moemoechu = 2
latent_size = 32
beta_trans = 4
joint_num = 25
beta_predict = 0
predicted_size = None
predicted_sizes = None
input_size = None
input_sizes = None
num_frames = None
inputs_avg = None
inputs_std = None
inputs_avg_gpu = None
inputs_std_gpu = None

def add02av(angular_velocity):
    angular_velocity/=np.linalg.norm(angular_velocity)
    return np.concatenate([angular_velocity, np.asarray([0])], axis=-1)


class motion_data_set(torch.utils.data.Dataset):
    def __init__(self, data, root_info):
        self.dataset, self.root_ori, self.root_pos = [], [], []
        for i in range (data.shape[0] - clip_size):
            datapiece = data[i:i+clip_size, :]
            datapiece = datapiece
            self.dataset.append (torch.tensor(datapiece).to(torch.float32).to(device).detach())
            self.root_ori.append (root_info[i+1][0])
            self.root_pos.append (root_info[i+1][1])
    
    def __getitem__(self, index):
        return self.dataset[index], self.root_ori[index], self.root_pos[index]
    
    def __len__(self):
        return len(self.dataset)


def move_to_01 (data):
    avg,std = np.mean(data,axis=(0)), np.std(data,axis=(0))+0.1
    data = (data - avg) / std
    return data, avg, std
def move_input_to01(x,gpu=False):
    if(gpu):
        x = (x - inputs_avg_gpu) / inputs_std_gpu
    else:  
        x = (x - inputs_avg) / inputs_std
    return x
def move_input_from01(x,gpu=False):
    if(gpu):
        x = x * inputs_std_gpu + inputs_avg_gpu
    else:
        x = x * inputs_std + inputs_avg
    return x



def transform_bvh (bvh, start=1):
    global predicted_size, predicted_sizes, input_size, input_sizes, num_frames
    #assume root is at index 0
    bvh._joint_position, bvh._joint_orientation = None,None
    bvh = bvh.recompute_joint_global_info()
    
    linear_velocity = bvh.compute_linear_velocity (False)
    angular_velocity = bvh.compute_angular_velocity (False)
    position, orientation = 1*bvh._joint_position, 1*bvh._joint_orientation
    
    motions, translations = [], []
    num_frames = bvh.num_frames
    for i in range(start, bvh.num_frames):
        motion, translation = [], []
        root_ori = orientation[i, 0, :]
        root_ori2 = R(root_ori).inv()
        
        for j in range(1, len(bvh._skeleton_joints)):
            translation.append (root_ori2.apply(position[i, j] - position[i, 0]))
            motion.append (quat_product(root_ori, orientation[i, j], inv_p=True))
        
        for j in range(0, len(bvh._skeleton_joints)):
            if (j == 0):
                translation.append (linear_velocity[i, j])
                motion.append (angular_velocity[i, j])
            else:
                translation.append (root_ori2.apply(linear_velocity[i, j]))
                motion.append (root_ori2.apply(angular_velocity[i, j]))
            if (predicted_size == None):
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
    
    print(predicted_sizes, input_sizes)
    
    root_info = [(1*orientation[i, 0, :], 1*position[i, 0, :]) for i in range(0, num_frames)]
    
    return motions, translations, root_info

def calc_root_ori(root_ori, angular_velocity, bvh):
    angular_velocity = quat_product(root_ori,add02av(angular_velocity))/2
    #angular_velocity = R.from_rotvec(angular_velocity).as_quat()/2
    v = root_ori + angular_velocity / bvh._fps 
    v = v / np.linalg.norm(v)
    return v


'''
def compute_motion_info (x, root_pos, root_ori, jtb, jrb, bvh):
    joint_position, joint_orientation = [root_pos], [root_ori]
    print(root_ori)
    root_ori2 = R(root_ori)
    bs = input_sizes[0]
    
    for i in range(0, joint_num - 1):
        joint_position.append(root_ori2.apply(x[bs+i*3:bs+i*3+3]) + root_pos)
        joint_orientation.append(quat_product(root_ori, x[i*4:i*4+4]))
        
    joint_translation, joint_rotation = None, None
    joint_translation, joint_rotation =\
        bvh.compute_joint_local_info ([joint_position], [joint_orientation], joint_translation, joint_rotation)
    return joint_translation[0], joint_rotation[0]
'''
def compute_motion_info (x, root_pos, root_ori, jtb, jrb, bvh):
    jt, jr = [root_pos], [root_ori]
    root_ori2 = R(root_ori)
    bs = input_sizes[0]+predicted_sizes[1]
    bt = predicted_sizes[0]
    for i in range(0, joint_num - 1):
        jt.append(root_ori2.apply(x[bs+i*3:bs+i*3+3])/ bvh._fps + jtb[i+1])

        jr.append(calc_root_ori(jrb[i+1],root_ori2.apply(x[bt+i*3:bt+i*3+3]),bvh))
    return jt, jr

def transform_root_from_input (x, root_ori_b, root_pos_b, bvh):
    ori, pos = x [:predicted_sizes[0]], x[input_sizes[0]:input_sizes[0]+predicted_sizes[1]]
    root_ori = calc_root_ori(root_ori_b, ori[-3:], bvh)
    root_pos = root_pos_b + pos[-3:]/bvh._fps
    return root_ori, root_pos

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("train model on device:" + str(device))
    
    make_new = False
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'new'):
            make_new =  True
    
    
    bvh = BVHLoader.load (data_path)
    bvh = bvh.sub_sequence(400)
    print("read " + str(bvh.num_frames) + " motions from " + data_path)
    motions, translations, root_info = transform_bvh(bvh)
    inputs, inputs_avg, inputs_std = move_to_01 (np.concatenate([motions,translations],axis=-1))
    inputs_avg_gpu, inputs_std_gpu = torch.tensor(inputs_avg).to(torch.float32).detach().to(device), torch.tensor(inputs_std).to(torch.float32).detach().to(device)
    assert (input_size == inputs.shape [1])
    
    train_motions, test_motions = train_test_split(inputs, test_size = 0.01)
    
    
    encoder = model.VAE_encoder (input_size, used_motions, h1, h2, latent_size)
    decoder = model.VAE_decoder (input_size, used_motions, latent_size, input_size, h2, h1, moemoechu)
    
    VAE = model.VAE(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr = learning_rate)
    iteration = 300
    epoch = 0
    p0_iteration, p1_iteration = 60, 20
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
        drop_last = True,\
        shuffle = False)
    test_loader = torch.utils.data.DataLoader(\
        dataset = motion_data_set (test_motions, root_info),\
        batch_size = batch_size,\
        drop_last = True,\
        shuffle = False)
    
    loss_MSE = torch.nn.MSELoss(reduction = 'sum')
    loss_KLD = lambda mu,logvar : -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)/logvar.numel()
       # lambda mu,sigma:-0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)



    while (epoch < iteration):
        teacher_p = (p0_iteration - epoch) / (p0_iteration - p1_iteration)
        epoch += 1 
        
        t = tqdm (train_loader, desc = f'[train]epoch:{epoch}')
        train_loss, train_nsample = 0, 0
        tot_loss_re, tot_loss_norm, tot_loss_moe, tot_loss_para = 0,0,0,0
        tot_loss_re1, tot_loss_re2 = 0,0
        
        #beta_VAE2 = beta_VAE
        beta_VAE2 = beta_VAE / beta_grow_round * (divmod(epoch, beta_grow_round)[1])
        for motions, root_ori, root_pos in train_loader:
            gt = move_input_from01(motions.detach(),True)
            gt = gt.detach()
            x = motions [:, 0, :]

            for i in range (1, clip_size):
                re_x, mu, sigma, moe_output = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
                
                loss_re1 = loss_MSE(re_x, gt [:, i, :])
                k1 = slice(predicted_sizes[0]-3, input_sizes[0], 1)
                k2 = slice(input_sizes[0]+predicted_sizes[1]-3, input_size, 1)
                loss_re2 = loss_MSE(re_x[:, k1], gt[:, i, k1]) + loss_MSE(re_x[:, k2], gt[:, i, k2])
                loss_re = loss_re1 * (1-beta_predict) + loss_re2 * beta_predict
                #loss_re = torch.tensor(0)
                loss_moe = 0
                
                moemoe, moemoepara = moe_output
                
                loss_norm = loss_KLD(mu, sigma)
                loss = loss_re + beta_VAE2 * loss_norm + beta_moe * loss_moe #+ beta_para * loss_para
           
                tot_loss_re += loss_re.item()
                tot_loss_norm += loss_norm.item()
                tot_loss_re1 += loss_re1.item()
                tot_loss_re2 += loss_re2.item()
                #tot_loss_moe += loss_moe.item()
                #tot_loss_para += loss_para.item()
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (random.random() < teacher_p):
                    x = motions[:, i, :].detach()
                else:
                    x = move_input_to01(re_x.detach(),True).detach()
                
                
            train_nsample += batch_size * (clip_size - 1)
            t.set_postfix({'loss':train_loss/train_nsample})
        '''
        #####-------revue-------#####
        #i dont want to admit that, but i do mix up the test and val part...
        tot_loss_re_val, tot_loss_norm_val = 0,0
        test_nsample = 0
        for motions, root_ori, root_pos in test_loader:
            x = motions [:, 0, :]
            for i in range (1, clip_size):
                re_x, mu, sigma, moe_output = VAE(torch.concat ([x, motions [:, i, :]], dim = 1))
                
                loss_re1 = loss_MSE(re_x, motions [:, i, :])
                k1 = slice(predicted_sizes[0]-3, input_sizes[0], 1)
                k2 = slice(input_sizes[0]+predicted_sizes[1]-3, input_size, 1)
                loss_re2 = loss_MSE(re_x[:, k1], motions [:, i, k1]) + loss_MSE(re_x[:, k2], motions [:, i, k2])
                loss_re = loss_re1 * (1-beta_predict) + loss_re2 * beta_predict
                loss_norm = loss_KLD(mu, sigma)

                tot_loss_re_val += loss_re.item()
                tot_loss_norm_val += loss_norm.item()
                
                x = re_x.detach()
            test_nsample += batch_size * (clip_size - 1)

               
                
        test_loss = (tot_loss_re_val+beta_VAE*tot_loss_norm_val)
        loss_history['test'].append(test_loss/test_nsample )
        ''' 
        writer.add_scalar(tag="loss_re",
            scalar_value=tot_loss_re/train_nsample,
            global_step=epoch)
        writer.add_scalar(tag="loss_norm",
            scalar_value=tot_loss_norm/train_nsample,
            global_step=epoch)
        writer.add_scalar(tag="loss_re1",
            scalar_value=tot_loss_re1/train_nsample,
            global_step=epoch)
        writer.add_scalar(tag="loss_re2",
            scalar_value=tot_loss_re2/train_nsample,
            global_step=epoch)
        '''
        writer.add_scalar(tag="loss_norm_val",
            scalar_value=tot_loss_norm_val/test_nsample,
            global_step=epoch)
        writer.add_scalar(tag="loss_re_val",
            scalar_value=tot_loss_re_val/test_nsample,
            global_step=epoch)
        '''
                  
        state = {'model': VAE.state_dict(),\
                    'epoch': epoch,\
                    'loss_history': loss_history,\
                    'optimizer': optimizer,\
                    'input_size': input_size,\
                    'input_sizes': input_sizes,\
                    'predicted_size': predicted_size,\
                    'predicted_sizes': predicted_sizes,\
                    'inputs_std': inputs_std,\
                    'inputs_avg': inputs_avg,\
                    }
        torch.save(state, output_path+'/final_model.pth')
        print ("iteration %d/%d, train_loss: %f, test_loss: %f", epoch, iteration, train_loss/train_nsample,\
            0)#test_loss/test_nsample)
        
    
    
'''
def transform_as_predict (o):
    return [o[:, :predicted_sizes[0]],\
        o[:, input_sizes[0]:input_sizes[0] + predicted_sizes[1]]]


def compute_joint_orientation (x, root_ori, root_pos, bvh, bs):
    joint_position, joint_orientation = [root_pos], [root_ori]
    
    root_ori2 = R(root_ori)
    
    for i in range(0, joint_num - 1):
        joint_position.append(root_ori2.apply(x[bs+i*3:bs+i*3+3]) + root_pos)
        joint_orientation.append((root_ori2 * R(x[i*4:i*4+4])).as_quat())
    return joint_orientation


def transform_as_input (x, re_x, root_ori_b, root_pos_b, bvh):
    def compute_angular_velocity(_joint_orientation):
        qd = np.diff(_joint_orientation, axis=0) * bvh._fps

        q = _joint_orientation[:-1] #if forward else self._joint_orientation[1:]
        q_conj = q.copy().reshape(-1, 4)
        q_conj[:, :3] *= -1
        qw = quat_product(qd.reshape(-1, 4), q_conj)

        w = np.zeros((2, bvh._num_joints, 3))
        frag = 2 * qw[:, :3].reshape(1, bvh._num_joints, 3)
        w[1:] = frag

        w[0] = w[1]
        return w
        
    ori, pos = re_x[:predicted_sizes[0]], re_x[predicted_sizes[0]:]
    root_ori2 = calc_root_ori(root_ori_b, ori[-3:], bvh)
    root_pos2 = root_pos_b + pos[-3:]/bvh._fps
    
    root_ori3 = R(root_ori2)
    root_ori3 = root_ori3.inv()
    extra_t, extra_r = [], []
    
    
    angular_velocity = compute_angular_velocity (np.asarray([\
        compute_joint_orientation(x, root_ori_b, root_pos_b, bvh, input_sizes[0]),\
        compute_joint_orientation(re_x, root_ori2, root_pos2, bvh, predicted_sizes[0])]))

    rdif = (root_pos2 - root_pos_b)
    for j in range(0, joint_num):
        jrd = angular_velocity [1, j]
        
        if(j==0):
            jtd = rdif * bvh._fps
            extra_r.append(torch.tensor(jrd))
            extra_t.append(torch.tensor(jtd))
        else:
            x_bs, re_x_bs = input_sizes[0]+j*3, predicted_sizes[0]+j*3
            jtd = (re_x[re_x_bs:re_x_bs+3] - x[x_bs:x_bs+3] + rdif) * bvh._fps
            extra_r.append(torch.tensor(((root_ori3 * R.from_rotvec(jrd)).as_rotvec())))
            extra_t.append(torch.tensor((root_ori3.apply(jtd))))
    
    return torch.concat ([torch.tensor(ori[:-3])] + extra_r +\
                         [torch.tensor(pos[:-3])] + extra_t, dim = -1)


def transform_root (re_x, root_ori_b, root_pos_b, bvh):
    ori, pos = re_x [:predicted_sizes[0]], re_x[predicted_sizes[0]:]
    root_ori = calc_root_ori(root_ori_b, ori[-3:], bvh)
    root_pos = root_pos_b + pos[-3:]/bvh._fps
    return root_ori, root_pos
'''
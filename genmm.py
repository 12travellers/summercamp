import torch
import pymotionlib
from pymotionlib import BVHLoader
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import unfoldNd
from pymotionlib.Utils import quat_product
import torch.functional as F


output_path = "./output"
data_path = "./walk1_subject5.bvh"

inputs_avg = None
inputs_std = None

def matrix_to_r6(matrix: torch.Tensor) -> torch.Tensor:
    matrix = torch.tensor(matrix)
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
def r6_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    d6 = torch.Tensor(d6)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def add02av(angular_velocity):
    angular_velocity/=np.linalg.norm(angular_velocity)
    return np.concatenate([angular_velocity, np.asarray([0])], axis=-1)

def move_to_01 (data):
    avg,std = np.mean(data,axis=(0)), np.std(data,axis=(0))+0.1
    data = (data - avg) / std
    return data, avg, std
def move_input_to01(x):
    x = (x - inputs_avg) / inputs_std
    return x
def move_input_from01(x):
    x = x * inputs_std + inputs_avg
    return x


JRS = None
def transform_bvh (jt, jr):
    global JRS
    JRS = jr.shape
    jrn=[]
    for i in range(JRS[0]):
        jrnn=[]
        for j in range(JRS[1]):
            jrnn.append(matrix_to_r6(R(jr[i,j]).as_matrix()).numpy().reshape(1,6))
        jrn.append(np.stack(jrnn, axis=0))
    
            
    return np.concatenate (np.asarray(jrn), jt, axis=-1)
def transform_output (output):
    bs = JRS.shape[-1]*6
    jrnn, jt = output[:, :, :bs],output[:, :, bs:]
    jr, jt = [], []
    for i in range(JRS[0]):
        jrn, jtn =[], []
        for j in range(JRS[1]):
            jrn.append(R.from_matrix(r6_to_matrix(jrnn[i,j]).numpy()).as_quat().reshape(1,4))
        jr.append(np.stack(jrn, axis=0))
    
    return jr,jt

def calc_root_ori(root_ori, angular_velocity, bvh):
    angular_velocity = quat_product(root_ori,add02av(angular_velocity))/2
    #angular_velocity = R.from_rotvec(angular_velocity).as_quat()/2
    v = root_ori + angular_velocity / bvh._fps 
    v = v / np.linalg.norm(v)
    return v

def get_initial(length, inputs):
    initial_motion = torch.randn((length, inputs.shape[1], inputs.shape[2]))
    #initial_motion += F.interpolate(torch.tensor(inputs[0]) ,size = length, mode='linear', align_corners=True)
    return torch.fmod(initial_motion, 1.0)

def extract_patches(x, patch_size=7, stride=1):
    assert(len(x.shape) == 3)
    b, c, _t = x.shape
    x_patches = unfoldNd.unfoldNd(x.transpose(1, 2), kernel_size=patch_size, stride=stride)
    return x_patches.reshape(b,-1,c*patch_size)

def naiveDistance(x, y):
    return np.sum((extract_patches(x)-extract_patches(y))**2,axis=-1), extract_patches(y)
def naiveBlend(output, ys, patch_size=7, stride=1):
    b, c, d = output.shape
    ys=ys.reshape(b, c, d)
    combined = unfoldNd.foldNd(ys.permute(0, 2, 1), output_size=(d), kernel_size=patch_size, stride=stride)
    input_ones = torch.ones_like(output, dtype=ys.dtype, device=ys.device)
    divisor = unfoldNd.unfoldNd(input_ones, kernel_size=patch_size, stride=stride)
    divisor = unfoldNd.foldNd(divisor, output_size=(d), kernel_size=patch_size, stride=stride)
    return combined / divisor
    
    
def transform(output, inputs, length, distance, blend):
    distances, patches = [], []
    for i in range(inputs):
        input = F.interpolate(inputs[i], size=length, mode='linear', align_corners=False)
        dist, patch = distance(output, input)
        assert(len(dist.shape)==1)
        distances.append(dist)
        patches.append(patch)
    distances, patches = np.stack(distances, axis=0), np.stack(patches, axis=0)
    return blend(output, patches.permute(1,0)[enumerate(np.argmin(distances.permute(1,0), axis=1))])
    
if __name__ == '__main__':
    
    make_new = False
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'new'):
            make_new =  True
    
    
    BVH = BVHLoader.load (data_path)
    samples = [(400,1800),(2000,2500),(3200,3900),(6700,8700)]
    inputs = []
    for start,end in samples:
        bvh = BVH.sub_sequence(start, end)
        print("selectivly read " + str(bvh.num_frames) + " motions from " + data_path)
        input = transform_bvh(bvh._joint_translation, bvh._joint_rotation)
        inputs.append(input)
    
    inputs = np.asarray(inputs) #[frames, movements, representations]
    #inputs, inputs_avg, inputs_std = move_to_01(inputs)
    
    print(bvh.joint_names)
    '''
    ['RootJoint', 'lHip', 'lKnee', 'lAnkle', 
    'lToeJoint', 'lToeJoint_end', 'rHip', 'rKnee', 
    'rAnkle', 'rToeJoint', 'rToeJoint_end', 'pelvis_lowerback', 
    'lowerback_torso', 'torso_head', 'torso_head_end', 'lTorso_Clavicle', 
    'lShoulder', 'lElbow', 'lWrist', 'lWrist_end', 
    'rTorso_Clavicle', 'rShoulder', 'rElbow', 'rWrist', 
    'rWrist_end']
    '''
    
    skeletons = \
            [[0,1,6,11,12],\
            [1,2,3,4,5],\
            [7,8,9,10],\
            [12,13,14,15,20],\
            [15,16,17,18,19],\
            [20,21,22,23,24]]
    
    
    
    initial_length, final_length, ratio = 100, 1000, 0.8
    output = get_initial (initial_length, inputs)
    while len(output) < final_length:
        length = len(output)
        target_length = round(length/ratio)
        if(target_length == length):
            target_length += 1
        for i in range(5):
            output = transform(output, inputs, target_length, naiveDistance, naiveBlend)
    
    
    
    sp = BVH.num_frames
    #output = move_input_from01(output)
    jr, jt = transform_output(output)
    BVH.append_trans_rotation(jt, jr)
    
    BVHLoader.save(bvh.sub_sequence(sp, BVH.num_frames), './infered.bvh')
    
    
    os.system("python -m pymotionlib.editor")
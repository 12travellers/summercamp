import torch
import pymotionlib
from pymotionlib import BVHLoader
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import unfoldNd
from pymotionlib.Utils import quat_product
import torch.nn.functional as F


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
        rot = R(jr[i][0])
        for j in range(JRS[1]):
            if(j!=0):
                jr[i][j]=(rot.inv()*R(jr[i][j])).as_quat()
                jt[i][j]-=jt[i][0]
            jrnn.append(matrix_to_r6(R(jr[i,j]).as_matrix()).numpy())
        jrn.append(np.stack(jrnn, axis=0))
    
    print(np.asarray(jrn).shape)
    return np.concatenate ([np.asarray(jrn), jt], axis=-1)
def transform_output (output):
    jrnn, jt = output[:, :, :6],output[:, :, 6:]
    jr = []
    for i in range(jrnn.shape[0]):
        jrn = []
        rot = None
        for j in range(JRS[1]):
            q = R.from_matrix(r6_to_matrix(jrnn[i,j,:]).numpy()).as_quat()
            if(j!=0):
                jt[i][j]+=jt[i][0]
                q=(rot*R(q)).as_quat()
            else:
                rot = R(q)
            jrn.append(q)
        jr.append(np.stack(jrn, axis=0))
    return jr,jt

def calc_root_ori(root_ori, angular_velocity, bvh):
    angular_velocity = quat_product(root_ori,add02av(angular_velocity))/2
    #angular_velocity = R.from_rotvec(angular_velocity).as_quat()/2
    v = root_ori + angular_velocity / bvh._fps 
    v = v / np.linalg.norm(v)
    return v

def get_initial(length, inputs):
    initial_motion = torch.randn((length, inputs[0].shape[1], inputs[0].shape[2]))
    #initial_motion += F.interpolate(torch.tensor(inputs[0]) ,size = length, mode='linear', align_corners=True)
    return torch.fmod(initial_motion, 1.0)

def extract_patches(x, patch_size=7, stride=1):
    assert(len(x.shape) == 3) #[frames, joints, representations]
    b, c, d = x.shape
    x_patches = unfoldNd.unfoldNd(torch.tensor(x).permute(1,2,0), kernel_size=patch_size, stride=stride).permute(2,0,1).numpy()
    return x_patches.reshape(b-patch_size+1,-1), x_patches.reshape(b-patch_size+1,-1,c*patch_size)

def naiveDistance(x, y):
    return np.sum((extract_patches(x)[0]-extract_patches(y)[0])**2,axis=-1), extract_patches(y)[1]
def naiveBlend(output, ys, patch_size=7, stride=1):
    b, c, d = output.shape
    print(b,c,d)
    ys=np.asarray(ys).reshape(b-patch_size+1, c, patch_size*d)
    print(torch.Tensor(ys).permute(1,2,0).shape)
    combined = unfoldNd.foldNd(torch.Tensor(ys).permute(1,2,0), output_size=(b,), kernel_size=patch_size, stride=stride)
    print(combined.shape)
    input_ones = torch.ones_like(combined, dtype=torch.tensor(ys).dtype)
    divisor = unfoldNd.unfoldNd(input_ones, kernel_size=patch_size, stride=stride)
    divisor = unfoldNd.foldNd(divisor, output_size=(b,), kernel_size=patch_size, stride=stride)
    print(divisor.shape)
    return (combined / divisor).permute(2,0,1).numpy()
    
    
def transform(output, inputs, length, distance, blend):
    distances, patches = [], []
    for i in range(len(inputs)):#[frames, joints, representations]
        input = F.interpolate(torch.tensor(inputs[i]).permute(1,2,0), size=length,\
            mode='linear', align_corners=False).permute(2,0,1).numpy()
        
        dist, patch = distance(output, input)
        assert(len(dist.shape)==1)
        distances.append(dist)
        patches.append(patch)
    distances, patches = np.stack(distances, axis=0), np.stack(patches, axis=0)
    print(distances.shape, patches.shape)
    nearest = list(enumerate(np.argmin(distances.transpose(1,0), axis=1)))
    print(patches.shape)
    print(nearest)
    #debug
    return blend(output, [patches[j,i] for (i,j) in nearest])
    
if __name__ == '__main__':
    
    make_new = False
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'new'):
            make_new =  True
    
    
    BVH = BVHLoader.load (data_path)
    samples = [(400,1400),(2000,3000),(3200,4200),(6700,7700)]
    inputs = []
    for i,(start,end) in enumerate(samples):
        bvh = BVH.sub_sequence(start, end)
        print("selectivly read " + str(bvh.num_frames) + " motions from " + data_path)
        input = transform_bvh(bvh._joint_translation, bvh._joint_rotation)
        print(input.shape)
        
        inputs.append(input)
        print(input)
        BVHLoader.save(bvh, './sample'+str(i)+'.bvh')
        
    
    inputs #[ [frames, joints, representations], ...]
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
    
    
    initial_length, final_length, ratio = 100, 1000, 0.7
    output = get_initial (initial_length, inputs)

    while output.shape[0] < final_length:
        length = output.shape[0]
        target_length = round(length/ratio)
        if(target_length == length):
            target_length += 1
            
        output = F.interpolate(torch.tensor(output).permute(1,2,0),\
            size=target_length, mode='linear', align_corners=False).permute(2,0,1).numpy()

        for i in range(2):
            output = transform(output, inputs, target_length, naiveDistance, naiveBlend)
    
    sp = BVH.num_frames
    #output = move_input_from01(output)
    jr, jt = transform_output(output)
    print(np.asarray(jt).shape,np.asarray(jr).shape)
    BVH.append_trans_rotation(np.asarray(jt), np.asarray(jr))
    
    BVHLoader.save(BVH.sub_sequence(sp, BVH.num_frames), './infered.bvh')
    
    
    os.system("python -m pymotionlib.editor")
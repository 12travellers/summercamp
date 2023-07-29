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
BVH = None

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
BS = None
def transform_bvh (jt, jr):
    jp, jo = None, None
    jp, jo = BVH.compute_joint_global_info(jt,jr,jp,jo)
    global JRS
    JRS = jr.shape
    jrn, rts =[], []
    for i in range(JRS[0]):
        jrnn=[]
        rts.append(np.concatenate([
                   matrix_to_r6(R(jo[i][0]).as_matrix()).numpy(),
                    jp[i][0]-(jp[i-1][0] if i>0 else 0)],axis=-1))
        rot = R(jo[i][0])
        for j in range(1, JRS[1]):
            jo[i][j]=(rot.inv()*R(jo[i][j])).as_quat()
            jp[i][j]-=jp[i][0]
            jp[i][j]=rot.inv().apply(jp[i][j])
            jrnn.append(matrix_to_r6(R(jo[i,j]).as_matrix()).numpy())
        jrn.append(np.stack(jrnn, axis=0))
    return np.transpose(np.concatenate ([np.asarray(rts),np.asarray(jrn).reshape(JRS[0], -1), jp.reshape(JRS[0], -1)], axis=-1), (1, 0))

def transform_output (output):
    output = output.transpose(1, 0)
    jp, jo = [], []
    for i in range(output.shape[0]):
        jp.append([])
        jo.append([])
        rot = R.from_matrix(r6_to_matrix(output[i,:6]).numpy())
        jo[i].append(rot.as_quat())
        output[i,6:9]+=(output[i-1,6:9] if i>0 else 0)
        jp[i].append(output[i,6:9])
        for j in range(0,JRS[1]-1):
            q = R.from_matrix(r6_to_matrix(output[i,9+j*6:15+j*6]).numpy()).as_quat()
            q=(rot*R(q)).as_quat()
            jo[i].append(q)
        another = (JRS[1]-1)*6+9
        for j in range(0,JRS[1]-1):
            p = output[i,another+j*3:another+3+j*3]
            jp[i].append(rot.apply(p+jp[i][0]))
    jt, jr = None, None      
    jt, jr = BVH.compute_joint_local_info(np.asarray(jp), np.asarray(jo), jt, jr)
    return jr,jt

def calc_root_ori(root_ori, angular_velocity, bvh):
    angular_velocity = quat_product(root_ori,add02av(angular_velocity))/2
    #angular_velocity = R.from_rotvec(angular_velocity).as_quat()/2
    v = root_ori + angular_velocity / bvh._fps 
    v = v / np.linalg.norm(v)
    return v

def get_initial(length, inputs):
    initial_motion = torch.randn((inputs[0].shape[0], length))
    for input in inputs: 
        initial_motion += F.interpolate(torch.tensor(input).unsqueeze(0),\
            size=length, mode='linear', align_corners=False).squeeze(0).numpy()
    return torch.fmod(initial_motion, 1.0)

def extract_patches(x, patch_size=7, stride=1):
    assert(len(x.shape) == 2) #[representations, frames]
    b, c = x.shape
    x_patches = unfoldNd.unfoldNd(torch.tensor(x).unsqueeze(0), kernel_size=patch_size, stride=stride).squeeze(0).numpy()
    x_patches = x_patches.transpose(1,0)
    print(x.shape, x_patches.shape)
    assert(x_patches.shape[1]==patch_size * b)
    return x_patches #[whatever, patch_size * frames]

def naiveDistance(x, y):
    return np.sum((x-y)**2,axis=-1)
def naiveBlend(output, ys, patch_size=7, stride=1):
    b, c = output.shape #[representations, frames]
    ys = ys.transpose(1,0)
    print("blending on:",b,c,ys.shape)
    combined = unfoldNd.foldNd(torch.Tensor(ys).unsqueeze(0), output_size=(c,),\
        kernel_size=patch_size, stride=stride)
    print(combined.shape)
    input_ones = torch.ones_like(combined, dtype=torch.tensor(ys).dtype)
    divisor = unfoldNd.unfoldNd(input_ones, kernel_size=patch_size, stride=stride)
    divisor = unfoldNd.foldNd(divisor, output_size=(c,), kernel_size=patch_size, stride=stride)
    
    return (combined / divisor).squeeze(0).numpy()
    
    
def transform(output, inputs, length, distance, blend):
    print("-----trans---------")
    print(output.shape, length)
    patches = []
    for i in range(len(inputs)):#[representations, frames]
        input = F.interpolate(torch.tensor(inputs[i]).unsqueeze(0), size=length,\
            mode='linear', align_corners=False).squeeze(0).numpy()
        patches.append(extract_patches(input)) #[whatever, patch_size * frames]
    
    patches = np.concatenate(patches, axis=0)
    xs = extract_patches(output)
    nearest = []
    for i in range(xs.shape[0]):
        j = np.argmin([distance(xs[i], patches[j]) for j in range(patches.shape[0])])
        nearest.append(j)
    
    return blend(output, patches[nearest])

def start_from_0(x):
    return x
    ds = x[0, 0, :].copy()
    x[:, :, :] -= ds
    return x
if __name__ == '__main__':
    
    make_new = False
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'new'):
            make_new =  True
    
    
    BVH = BVHLoader.load (data_path)
    BVH = BVH.resample(60)
    samples = [(400,1000), (5000,5600), (8000,8600),(9000,9600), (4100,4700)]
    inputs = []
    for i,(start,end) in enumerate(samples):
        bvh = BVH.sub_sequence(start, end, copy=True)
        #bvh._joint_translation = start_from_0(bvh._joint_translation)
        print("selectivly read " + str(bvh.num_frames) + " motions from " + data_path)
        input = transform_bvh(bvh._joint_translation.copy(), bvh._joint_rotation.copy())
        print(input.shape)
        
        inputs.append(input)
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
    
    
    initial_length, final_length, ratio = 60, 1000, 0.75
    output = get_initial (initial_length, inputs)
    
    while output.shape[1] < final_length:
        length = output.shape[1]
        target_length = round(length/ratio)
        if(target_length == length):
            target_length += 1
        
        output = F.interpolate(torch.tensor(output).unsqueeze(0),\
            size=target_length, mode='linear', align_corners=False).squeeze(0).numpy()
        print(output.shape)
        for i in range(3):
            output = transform(output, inputs, target_length, naiveDistance, naiveBlend)
    
    
    for shots in range(0,5):
        sp = BVH.num_frames
        #output = move_input_from01(output)
        jr, jt = transform_output(output)
        jt = start_from_0(jt)
        print(np.asarray(jt).shape,np.asarray(jr).shape)
        BVH.append_trans_rotation(np.asarray(jt), np.asarray(jr))
        
        BVHLoader.save(BVH.sub_sequence(sp, BVH.num_frames), './infered'+str(shots)+'.bvh')
    
    
    os.system("python -m pymotionlib.editor")
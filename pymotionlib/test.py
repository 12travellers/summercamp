'''
一个简单样例，在这里我们会将两个bvh文件融合，取其中一个的上半身，另一个的下半身，然后保存为一个新的bvh文件
'''

from pymotionlib import BVHLoader
import numpy as np

# 处理以下两个bvh
file_name_list = [r'0018_Xinjiang002-mocap-100.bvh', r'0018_Walking001-mocap-100.bvh']

bvh_list = []
for file_name in file_name_list:
    bvh_list.append(BVHLoader.load(file_name)) # BVHLoader.load会将bvh读进来，并转为MotionData类

name_list_1 = bvh_list[0]._skeleton_joints # _skeleton_joints 是名字列表
name_list_2 = bvh_list[1]._skeleton_joints

# map namelist 1 to namelist 2
index = []
for name in name_list_1:
    index.append(name_list_2.index(name)) # 这里是为了防止两个bvh关节顺序对不上

upper_body_mask = np.zeros(len(name_list_2)) # 这里用来标志每个关节是不是upper body
upper_body_mask[name_list_1.index('pelvis_lowerback')] = 1 # 认为所有pelvis_lowerback的儿子节点都是upper body

# 洪水染色
for i in range(1, len(name_list_1)):
    if upper_body_mask[bvh_list[0]._skeleton_joint_parents[i]] == 1:
        upper_body_mask[i] = 1

upper_body_name = [bvh_list[0]._skeleton_joints[i] for i in range(len(name_list_1)) if upper_body_mask[i] == 1]
print(upper_body_name)

num_frames = min(bvh_list[0].num_frames, bvh_list[1].num_frames)
bvh_list = [x.sub_sequence(0, num_frames) for x in bvh_list]
assert bvh_list[0].num_frames == bvh_list[1].num_frames

for i in range(len(name_list_1)):
    if upper_body_mask[i] == 1:
        bvh_list[1]._joint_rotation[:bvh_list[0].num_frames, index[i], :] = bvh_list[0]._joint_rotation[:, i, :]
        # _joint_rotation是bvh的局部旋转信息，形状是(num_frames, num_joints, 4)
        
bvh_list[1]._joint_rotation = np.ascontiguousarray(bvh_list[1]._joint_rotation)
BVHLoader.save(bvh_list[1], 'test.bvh')
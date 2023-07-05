import model
import pytorch
import pymotionlib




def read_motion_data (file_name_list):
    bvh_list = []
    for file_name in file_name_list:
        bvh_list.append(BVHLoader.load(file_name)) # BVHLoader.load会将bvh读进来，并转为MotionData类
    return bvh_list



if __name__ == 'main':
    motions = read_motion_data ([".\walk1_subject5.bvh"])
    
    data_loader = 
    
    output_path = 
    
    epoch = 
    for in range(epoch):
        
        

import torch
import pymotionlib
from pymotionlib import BVHLoader








if __name__ == '__main__':
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = "./output"
    bvh = BVHLoader.load (["./walk1_subject5.bvh"])
    print(bvh._joint_position)
    '''
    
    data_loader = 
    
    
    iteration = 10
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint(['epoch'])

    while (epoch < iteration):
        epoch += 1        
        state = {'model': model.state_dict(), \
                 'epoch': epoch}
        torch.save(state, output_path+'/final_model.pth')
    
    '''
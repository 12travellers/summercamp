import torch
import numpy as np


class VAE_encoder (torch.nn.Module):
    def __init__ (self, motion_size, used_motions, h1, h2, latent_size, used_angles):
        super (VAE_encoder, self).__init__()
        self.motion_size = motion_size
        self.used_motions = used_motions
        self.input_size = (motion_size + used_angles) * used_motions
        self.latent_size = latent_size
        self.used_angles = used_angles
        
        self.l1 = torch.nn.Linear (self.input_size, h1)
        self.l2 = torch.nn.Linear (h1, h2)
        
        self.mull = torch.nn.Linear (h2, latent_size)
        self.sigmall = torch.nn.Linear (h2, latent_size)
        
    def forward (self, motions, angles = torch.zeros(1, 0)):
        if (0 == self.used_angles):
            angles = torch.zeros(motions.shape[0], 0)
        input = torch.concat ([motions, angles], dim = 1)
        input = input.to(torch.float32)
        assert (input.shape[1] == self.input_size)
        input = torch.nn.functional.elu (self.l1 (input))
        input = torch.nn.functional.elu (self.l2 (input))
        return self.mull(input), self.sigmall(input)
        
        
class VAE_decoder (torch.nn.Module):
    def __init__ (self, motion_size, used_motions, latent_size, h1, h2, moe, used_angles):
        super (VAE_decoder, self).__init__()
        self.used_motions = used_motions
        self.motion_size = motion_size
        self.latent_size = latent_size
        self.moe = moe
        self.used_angles = used_angles
        self.input_size = (motion_size + used_angles) * (used_motions - 1) + latent_size
        
        
        self.l1 = torch.nn.ModuleList([torch.nn.Linear(self.input_size, h1) for i in range(self.moe)])
        self.l2 = torch.nn.ModuleList([torch.nn.Linear(h1 + latent_size, h2) for i in range(self.moe)])
        self.l3 = torch.nn.ModuleList([torch.nn.Linear(h2 + latent_size, motion_size) for i in range(self.moe)])

        self.gate1 = torch.nn.Linear (self.input_size, h1)
        self.gate2 = torch.nn.Linear (h1, h2)
        self.gate3 = torch.nn.Linear (h2, self.moe)

        
        
        
    def forward (self, motions, z, angles = torch.zeros(1, 0)):
        if (0 == self.used_angles):
            angles = torch.zeros(motions.shape[0], 0)
    
        input = torch.concatenate ([motions, z, angles], dim = 1)
        input = input.to(torch.float32)
    #    print(motions.shape, z.shape, self.input_size)
        assert (input.shape[1] == self.input_size)
        
        
        output = torch.zeros ([self.motion_size])
        
        para = torch.nn.functional.elu (self.gate1 (input)) 
        para = torch.nn.functional.elu (self.gate2 (para)) 
        para = torch.nn.functional.elu (self.gate3 (para)) 
        
        for i in range(self.moe):
            tmp = torch.nn.functional.elu (self.l1[i] (input))
            tmp = torch.nn.functional.elu (self.l2[i] (\
                torch.concatenate ([tmp, z], dim = 1)))
            tmp = torch.nn.functional.elu (self.l3[i] (\
                torch.concatenate ([tmp, z], dim = 1)))
            output = output + torch.mul (para[:, i:i+1], tmp)
        output = output / torch.norm (para).to (torch.float32)
        
        return torch.sigmoid(output)            
        
        

class VAE (torch.nn.Module):
    def __init__(self, encoder, decoder):
        super (VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, motions, angles = torch.zeros(1, 0)):
        mu, sigma = self.encoder (motions, angles)
        z = mu + torch.randn_like (sigma) * sigma
        
        re_build = self.decoder (motions[:, :-self.encoder.motion_size], z, angles[:, :-self.encoder.used_angles])
        return re_build, mu, sigma
    
    
if __name__ == '__main__':
    print("---revue starlight---")
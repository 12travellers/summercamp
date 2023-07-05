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
        self.sigmall = torch.nn.Linear (h2, latent_size,)
        
    def forward (self, motions, angles = np.zeros(0)):
        input = torch.concat ([motions, angles], dim = 1)
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
        self.used_angled = used_angles
        self.input_size = (motion_size + used_angles) * (used_motions - 1) + latent_size
        
        for i in range(self.moe):
            self.l1[i] = torch.nn.Linear (self.input_size, h1)
            self.l2[i] = torch.nn.Linear (h1, h2)
        self.para = torch.ones (self.moe)

        self.l3 = torch.nn.Linear (self.input_size + latent_size, h1)
        self.l4 = torch.nn.Linear (h1 + latent_size, h2)
        self.l5 = torch.nn.Linear (h2 + latent_size, motion_size)
        
        
    def forward (self, motions, z, angles = np.zeros(0)):
        input = torch.concatenate ([motions, angles, z], dim = 1)
        assert (input.shape[1] == self.input_size)
        
        
        output = torch.zeros (input.shape)
        inputs = torch.reshape (input, [1] + input.shape)
        inputs = torch.repeat_interleave(inputs, self.moe, dim = 0)
        
        for i in range(self.moe):
            inputs[i] = torch.nn.functional.elu (self.l1[i] (inputs[i]))
            inputs[i] = torch.nn.functional.elu (self.l2[i] (inputs[i]))
            output = output + self.para[i] * inputs[i]
        outupt = output / torch.norm (self.para)
        
        output = torch.nn.functional.elu (self.l3 (\
            torch.concatenate ([output, z], dim = 1)))
        output = torch.nn.functional.elu (self.l4 (\
            torch.concatenate ([output, z], dim = 1)))
        output = torch.nn.functional.elu (self.l5 (\
            torch.concatenate ([output, z], dim = 1)))
        return output            
        
        

class VAE (torch.nn.Module):
    def __init__(self, encoder, decoder):
        super (VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, motions, angles):
        mu, sigma = self.encoder (motions, angles)
        z = mu + torch.randn_like (sigma) * sigma
        re_build = self.decoder (motions[:, :-1], angles[:, :-1], z)
        return re_build, mu, sigma
    
    
if __name__ == '__main__':
    print("---revue starlight---")
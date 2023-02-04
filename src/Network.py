import os,pickle
import numpy as np
import torch
import torch.nn as nn

class opt:
    nc = 51 #num_channel
    hc = 32 #hidden channel 
    
def conv_block(inc,outc,k=4,s=2,p=1,b=False,bn=True):
    cblock = [] 
    cblock.append(
                  nn.Conv1d(inc,outc,k,s,p,bias=b)
                )
    if bn:
        cblock.append(nn.BatchNorm1d(outc))
    cblock.append(nn.LeakyReLU(0.2,inplace=True))
    return nn.Sequential(*cblock)

def conv_tp_block(inc,outc,k=4,s=2,p=1,b=False):
    return nn.Sequential(
        nn.ConvTranspose1d(inc,outc,k,s,p,bias=b),
        nn.BatchNorm1d(outc),
        nn.ReLU(True)
        )

class Encoder(nn.Module):
    def __init__(self,in_c,hidden_c,latent_c):
        super(Encoder, self).__init__()
        self.in_c = in_c # num of features of input data 
        self.hc = hidden_c
        self.layers = self.build_layers(latent_c)
        
    def build_layers(self,latent_c):
        layers = [] 
        for i,c in enumerate([0,1,2,4,8]):
            if i == 0:
                layers.append(conv_block(self.in_c,self.hc,bn=False))        
            else:
                layers.append(conv_block(self.hc*c,self.hc*c*2))
        layers.append(nn.Conv1d(self.hc * 16, latent_c, 10, 1, 0, bias=False))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        return self.layers(x)
    
class Decoder(nn.Module):
    def __init__(self,in_c,hidden_c,latent_c):
        super(Decoder,self).__init__()
        self.layers = nn.Sequential(
            conv_tp_block(latent_c,hidden_c*16,10,1,0),
            conv_tp_block(hidden_c*16,hidden_c*8),
            conv_tp_block(hidden_c*8,hidden_c*4),
            conv_tp_block(hidden_c*4,hidden_c*2),
            conv_tp_block(hidden_c*2,hidden_c*1),
            
            nn.ConvTranspose1d(hidden_c ,in_c, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self,x):
        return self.layers(x)
        
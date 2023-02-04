import torch.nn as nn 
import torch 
from src.Network import Encoder,Decoder
from torch import optim 
import torch.nn.functional as F 
import time 
import numpy as np 
from src.Network import Encoder,Decoder

class Generator(nn.Module):
    def __init__(self,in_c,hidden_c,latent_c):
        super(Generator,self).__init__()
        self.encoder = Encoder(in_c,hidden_c,latent_c)
        self.decoder = Decoder(in_c,hidden_c,latent_c)
    def forward(self,x):
        latent_z = self.encoder(x)
        recon_x = self.decoder(latent_z)
        return recon_x,latent_z
    
class Discriminator(nn.Module):
    def __init__(self,in_c,hidden_c,latent_c):
        super(Discriminator,self).__init__()
        self.layers = list(Encoder(in_c,hidden_c,latent_c).layers.children())
        self.encoder = nn.Sequential(*self.layers[:-1])
        self.classifier = nn.Conv1d(self.layers[-1].in_channels,1,10,1,0,bias=False)
        
    def forward(self,x):
        features = self.encoder(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1,1).squeeze(1)
        classifier = torch.sigmoid(classifier)
        return classifier,features 
        

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)

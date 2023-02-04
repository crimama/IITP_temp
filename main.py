import pandas as pd 
import os 
import numpy as np 
from glob import glob 
import argparse 
import yaml 

from src.datafactory import dataloader 
from src.model import Generator,Discriminator,weights_init
from src.Network import Encoder,Decoder 
from src.test import metrics 
from src.scheduler import CosineAnnealingWarmupRestarts

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn.metrics import roc_auc_score,roc_curve,auc 

class LossMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.loss_g_adv  = 0.0
        self.loss_g_rec  = 0.0
        self.loss_g      = 0.0 
        
        self.loss_d_real = 0.0 
        self.loss_d_fake = 0.0 
        self.loss_d      = 0.0 
        
        self.count = 0 
        
    def update(self,log,n=1):
        self.count += n 
        
        self.loss_g_adv  += log['loss_g_adv']
        self.loss_g_rec  += log['loss_g_rec']
        self.loss_g      += log['loss_g'] 
        
        self.loss_d_real += log['loss_d_real'] 
        self.loss_d_fake += log['loss_d_fake'] 
        self.loss_d      += log['loss_d'] 
        
    def avg(self):
        log = {
            'loss_g_adv'  : self.loss_g_adv/self.count,
            'loss_g_rec'  : self.loss_g_rec/self.count,
            'loss_g'      : self.loss_g    /self.count,
            'loss_d_real' : self.loss_d_real/self.count,
            'loss_d_fake' : self.loss_d_fake/self.count,
            'loss_d'      : self.loss_d    /self.count
            }
        self.reset()
        return log 
        


def run(cfg):
    device = cfg['TRAIN']['device']
    # build dataloader 
    trainloader = dataloader(
        datadir     = './Data/train_ver2.csv',
        window_size = cfg['DATA']['window_size'],
        stride      = cfg['DATA']['stride'],
        batch_size  = cfg['TRAIN']['batchsize'],
        shuffle     = True 
    )
    testloader = dataloader(
        datadir     = './Data/test_ver2.csv',
        window_size = cfg['DATA']['window_size'],
        stride      = cfg['DATA']['stride'],
        batch_size  = cfg['TRAIN']['batchsize'],
        shuffle     = False  
    ) 
    testloader.dataset.df['attack'] = testloader.dataset.df['attack'].apply(lambda x : round(x))
    
    # build model 
    D = Discriminator(
        cfg['MODEL']['in_c'],
        cfg['MODEL']['hidden_c'],
        cfg['MODEL']['latent_c']).apply(weights_init).to(device)
    G = Generator(
        cfg['MODEL']['in_c'],
        cfg['MODEL']['hidden_c'],
        cfg['MODEL']['latent_c']).apply(weights_init).to(device)
    
    # build loss function 
    bce_criterion = nn.BCELoss()
    mse_criterion = nn.MSELoss()

    # build optimizer 
    optimizerG = torch.optim.Adam(G.parameters(),lr=cfg['TRAIN']['lr'],betas=(0.999,0.999))
    optimizerD = torch.optim.Adam(D.parameters(),lr=cfg['TRAIN']['lr'],betas=(0.999,0.999))
    schedulerG = CosineAnnealingWarmupRestarts(
            optimizerG, 
            first_cycle_steps = cfg['TRAIN']['epochs'],
            max_lr            = cfg['TRAIN']['lr'],
            min_lr            = cfg['TRAIN']['min_lr'],
            warmup_steps      = int(cfg['TRAIN']['epochs'] * cfg['TRAIN']['warmup_ratio'])
        )
    schedulerD = CosineAnnealingWarmupRestarts(
            optimizerD, 
            first_cycle_steps = cfg['TRAIN']['epochs'],
            max_lr            = cfg['TRAIN']['lr'],
            min_lr            = cfg['TRAIN']['min_lr'],
            warmup_steps      = int(cfg['TRAIN']['epochs'] * cfg['TRAIN']['warmup_ratio'])
        )
    
    print('All loaded, train start')
    train(G,D,
          trainloader,testloader,
          optimizerG,optimizerD,
          schedulerG,schedulerD,
          bce_criterion,mse_criterion,
          device)
    
def train(G,
          D,
          trainloader,testloader,
          optimizerG,optimizerD,
          schedulerG,schedulerD,
          bce_criterion,mse_criterion,
          device):
    
    real_label = 1 #해당 라벨은 생성한 이미지가 진짜인지 가짜인지 판별하는 라벨 
    fake_label = 0 

    for epoch in range(cfg['TRAIN']['epochs']):
        #train epoch 
        G.train()
        D.train()
        
        # logging 
        loss_meter = LossMeter() 
        best_loss = np.inf
        
        #sceduler 
        schedulerG.step()
        schedulerD.step()
        for i,(x,y) in enumerate(trainloader):
            x,y = x.to(device),y.to(device)
            
            # ! update D 
            G.eval()
            D.train()
            D.zero_grad()
            
            # Train with real 
            out_d_real,_ = D(x)
            loss_d_real = bce_criterion(
                                        out_d_real,
                                        torch.full((x.shape[0],), real_label, device=device).type(torch.float32)
                                        )
            
            # Train with fake 
            with torch.no_grad():
                recon_x,_ = G(x)
            out_d_fake,_ = D(recon_x)
            loss_d_fake = bce_criterion(
                                        out_d_fake,
                                        torch.full((x.shape[0],),real_label,device=device).type(torch.float32)
                                        )
            
            # loss backward 
            loss_d = loss_d_real + loss_d_fake
            optimizerD.zero_grad()
            loss_d.backward()
            optimizerD.step()
            
            # ! update G 
            G.train()
            D.eval()
            G.zero_grad()
            
            #reconsturction 
            recon_x,latent_z = G(x)
            
            with torch.no_grad():
                _,feat_real = D(x) # original feature
                out_g, feat_fake = D(recon_x) # reconstruction feature 
            
            # loss 
            loss_g_adv = mse_criterion(feat_fake,feat_real) # loss for feature matching 
            loss_g_rec = mse_criterion(recon_x,x) # reconstruction 
            
            # backward 
            loss_g = loss_g_adv + loss_g_rec * 1 # w_adv = 1 
            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()
            
            #logging 
            log = {
            'loss_g_adv' : loss_g_adv ,
            'loss_g_rec' : loss_g_rec ,
            'loss_g'     : loss_g     ,
            'loss_d_real' : loss_d_real ,
            'loss_d_fake': loss_d_fake,
            'loss_d'     : loss_d     }
            loss_meter.update(log)
        
        # Epoch evaluate 
        log = loss_meter.avg()
        
        auroc,f1,y_true,y_pred,thr = metrics(G,testloader,device)
        print(f"Epoch:[{epoch}/{cfg['TRAIN']['epochs']:.3f}] |G loss : {log['loss_g']:.3f}|D Loss : {log['loss_d']:.3f}|AUROC : {auroc:.3f}|F1 : {f1:.3f}")
        if log['loss_g'] < best_loss:        
            torch.save(G,f"{cfg['SAVE']['savedir']}/best_G.pt")        
            torch.save(D,f"{cfg['SAVE']['savedir']}/best_D.pt")
        
    torch.save(G,f"{cfg['SAVE']['savedir']}/last_G.pt")        
    torch.save(D,f"{cfg['SAVE']['savedir']}/last_D.pt")
    
def init():
    # configs 
    parser = argparse.ArgumentParser(description='BeatGAN')
    parser.add_argument('--yaml_config', type=str, default='./configs/default.yaml', help='exp config file')    
    args = parser.parse_args()
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    
    # save point 
    savedir = os.path.join(cfg['SAVE']['savedir'],'Baseline')
    n = 0 
    while True:
        if os.path.exists(savedir+str(n)):
            n+=1 
        else:
            savedir = savedir + str(n)
            break 
    os.mkdir(savedir)
    cfg['SAVE']['savedir'] = savedir 
    
    with open(f"{savedir}/config.yaml",'w') as f:
            yaml.dump(cfg,f)
    return cfg 


if __name__ == '__main__':
    cfg = init() 
    run(cfg)
import torch 
from torch.utils.data import Dataset,DataLoader
import pandas as pd 
import numpy as np 
class SwatDataset(Dataset):
    
    def __init__(self,datadir,window_size,stride):
        super(SwatDataset,self).__init__()
        self.df = pd.read_csv(datadir).drop(columns='timestamp')
        self.df['attack'] = self.df['attack'].apply(lambda x : round(x))
        self.stride = stride 
        self.window = window_size 
        
    def __len__(self):
        return int((len(self.df)-self.window)/self.stride)
    
    def __getitem__(self,idx):
        values = self.df.iloc[idx*self.stride : idx*self.stride + self.window].values 
        x = values[:,:-1]
        #if label is last value of Y
        y = torch.max(torch.Tensor(values[:,-1]))
        #if label is max value of Y
        #y = values[:,-1]
        x = torch.Tensor(np.transpose(x,axes=(1,0))).type(torch.float32)
        return x,y  
        
def dataloader(datadir,window_size,stride,batch_size,shuffle):
    dataloader = DataLoader(
        dataset = SwatDataset(
                            datadir = datadir,
                            window_size = window_size,
                            stride = stride
                                ),
        batch_size = batch_size,
        shuffle = shuffle
    )
    return dataloader 
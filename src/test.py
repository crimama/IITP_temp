import torch 
import numpy as np 
from sklearn.metrics import roc_auc_score,roc_curve,auc ,accuracy_score,f1_score
import pandas as pd 
def min_max_norm(score_list):
    min_value = np.min(score_list)
    max_value = np.max(score_list)

    score_list = (np.array(score_list) - min_value) / (max_value-min_value)
    return score_list 

def metrics(G,testloader,device):
    y_list = [] 
    score_list = [] 
    G.eval()
    for x,y in testloader:
        x = x.to(device)
        
        with torch.no_grad():
            recon_x,_ = G(x)
        anomaly_score = torch.mean(torch.pow((x-recon_x),2),dim=(1,2)).detach().cpu().numpy()
        
        score_list.extend(anomaly_score)
        y_list.extend(y.detach().cpu().numpy())
    #min max norm 
    score_list = min_max_norm(score_list)
    
    #metrics 
    fpr,tpr,thr = roc_curve(y_list,score_list)
    auroc = auc(fpr,tpr)
    
    thres = np.percentile(score_list,80)
    f1 = f1_score(y_list,
                   pd.Series(score_list).apply(lambda x : 1 if x > thres else 0).values 
                   )
    
    y_true = y_list 
    y_pred = score_list 
    
    return auroc,f1,y_true,y_pred,thr 
        
import time ###
start = time.time() ###
import argparse 
import os
import numpy as np
import wandb
from datetime import timedelta

parser = argparse.ArgumentParser(description='training the model')   
parser.add_argument('-n', '--gpu', help='GPU number',type=str,required=True)
parser.add_argument('-f', '--file', help='save file',type=str,required=True)
parser.add_argument('-p', '--project_name', help='project_name',type=str,default='non')  
parser.add_argument('-hold', '--patience', help='early stop patience', type=int, default=10)
parser.add_argument('-train', '--bool_train', help='training? or test?', type=bool, default=False)
parser.add_argument('-cv', '--bool_cv', help='cross validation??', type=bool, default=False)
parser.add_argument('-fold', '--fold', help='dataset fold', type=int, default=0)
parser.add_argument('-data_name', '--data_name', help='data name',type=str,required=True)

args = parser.parse_args()  

print(args)

patience = 0


filesave = args.file
GPU_NUM =  args.gpu # Number of GPU

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM

# Setting Device
import torch
from torch_geometric.data import Batch
from scipy.stats import pearsonr
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from dataset import PDBbindDataset, BindingDBDataset
from models import Model
from dataset import load_data, load_label
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
print("CPU Random Seed :", torch.initial_seed())
print("CUDA Random Seed :", torch.cuda.initial_seed())
torch.autograd.set_detect_anomaly(True)

modelArgs = {}

modelArgs['pro_rep_dim'] = 1280
modelArgs['pk_rep_dim'] = 320
modelArgs['batch_size'] = 128
modelArgs['lstm_hid_dim'] = 64
modelArgs['d_a'] = 32
modelArgs['r'] = 768
modelArgs['dropout'] = 0.2
modelArgs['in_channels'] = 43
modelArgs['units_list'] = [128,64]
modelArgs['emb_dim'] = 512
modelArgs['dense_hid'] = 64
modelArgs['lr'] = 0.0001
modelArgs['lr_decay'] = 0.99
modelArgs['decay_interval'] = 10
modelArgs['weight_decay'] = 1e-6
modelArgs['iteration'] = 200
modelArgs['max_norm'] = 5
modelArgs['filesave'] = filesave
modelArgs['lambda_cont'] = 1000
modelArgs['lambda_l1'] = 0.001
model_num = '12'
modelArgs['T_max'] = 10
modelArgs['device'] = device

# print('modelArgs : ',modelArgs)

class Tester(object):
    def __init__(self, model, data_name):
        self.model = model
        self.data_name = data_name
    def test(self, dataloader):
        
        
        self.model.eval()
        N = len(dataloader)
        T = np.array([], dtype=float)
        Y = np.array([], dtype=float)
        
                
        for samples in dataloader:
            with torch.no_grad():        
                
                samples[0].seq = samples[0].seq.view(-1,1000,1280)
                samples[0].encode_input = samples[0].encode_input.view(-1,102)
                samples[0].encoder_pad_mask = samples[0].encoder_pad_mask.view(-1,102)
                
                if self.data_name=='pdbbind':
                    samples[0].pk = samples[0].pk.view(-1, 82) 
                elif self.data_name=='bindingdb':
                    samples[0].pk = samples[0].pk.view(-1, 434)
                else:
                    print('wrong data name')
                    
                (correct_scores, predicted_scores, loss) = self.model(samples, train=False)
                T = np.append(T,correct_scores)
                Y = np.append(Y,predicted_scores)
        if self.data_name=='pdbbind':
            pearson = CORR(T,Y)
            rmse = RMSE(T,Y)
            mae = MAE(T, Y)
            sd = SD(T, Y)
            ci = c_index(T, Y)
            return pearson, rmse, mae, sd, ci
        
        elif self.data_name=='bindingdb':
            pearson = CORR(T,Y)
            rmse = RMSE(T,Y)
            return pearson, rmse
        
        else:
            print('wrong data name')

    def save_rmse(self, rmse, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, rmse)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def collate_fn(batch):
    data, targets = zip(*batch)
    batched_data = Batch.from_data_list(data)
    batched_targets = torch.stack(targets)
    
    return batched_data, batched_targets


if args.data_name=='pdbbind':
    com_node, com_edge, com_padidx, com_token, com_pad, sequences, test195_pockets, test262_pockets, test290_pockets, test95_pockets = load_data(args.data_name, args.bool_cv, args.bool_train, args.fold)
    test2013_195_idx, test2013_95_idx, test2016_262_idx, test2016_290_idx = load_label(args.data_name, args.bool_cv, args.bool_train, args.fold)

    test2013_195_dataset = PDBbindDataset(idx=test2013_195_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=test195_pockets, batch_size=modelArgs['batch_size'])
    test2013_95_dataset = PDBbindDataset(idx=test2013_95_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=test95_pockets, batch_size=modelArgs['batch_size'])
    test2016_262_dataset = PDBbindDataset(idx=test2016_262_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=test262_pockets, batch_size=modelArgs['batch_size'])
    test2016_290_dataset = PDBbindDataset(idx=test2016_290_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=test290_pockets, batch_size=modelArgs['batch_size'])

    test2013_195_loader = torch.utils.data.DataLoader(dataset = test2013_195_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)
    test2013_95_loader = torch.utils.data.DataLoader(dataset = test2013_95_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)
    test2016_262_loader = torch.utils.data.DataLoader(dataset = test2016_262_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)
    test2016_290_loader = torch.utils.data.DataLoader(dataset = test2016_290_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)

    """Set a model."""
    model = Model(modelArgs)
    model.to(device)
    # PATH = './RESULT/'+'model--' + filesave+'.pth'
    PATH = '/NAS_Storage4/jaesuk/chem_esm_cont/src/RESULT/model--241226_PDBbind_dataset_lr_0.0001_hold_20_T_10.pth'
    model.load_state_dict(torch.load(PATH, map_location=device), strict=False)

    criterion = F.mse_loss
    tester = Tester(model, args.data_name)

    pearson, rmse, mae, sd, ci = tester.test(test2013_195_loader)
    print('test2013_195')
    print('pearson\trmse\tmae\tsd\tci')
    result = [round(pearson,6),round(rmse,6),round(mae,6),round(sd,6),round(ci,6)]
    print('\t '.join(map(str, result))) 

    pearson, rmse, mae, sd, ci = tester.test(test2013_95_loader)
    print('test2013_95')
    print('pearson\trmse\tmae\tsd\tci')
    result = [round(pearson,6),round(rmse,6),round(mae,6),round(sd,6),round(ci,6)]
    print('\t '.join(map(str, result))) 

    pearson, rmse, mae, sd, ci = tester.test(test2016_262_loader)
    print('test2016_262')
    print('pearson\trmse\tmae\tsd\tci')
    result = [round(pearson,6),round(rmse,6),round(mae,6),round(sd,6),round(ci,6)]
    print('\t '.join(map(str, result))) 

    pearson, rmse, mae, sd, ci = tester.test(test2016_290_loader)
    print('test2016_290')
    print('pearson\trmse\tmae\tsd\tci')
    result = [round(pearson,6),round(rmse,6),round(mae,6),round(sd,6),round(ci,6)]
    print('\t '.join(map(str, result))) 
    
elif args.data_name=='bindingdb':
    com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data = load_data(args.data_name, args.bool_cv, args.bool_train, args.fold)
    channel_idx, er_idx, gpcr_idx, kinase_idx = load_label(args.data_name, args.bool_cv, args.bool_train, args.fold)
    
    channel_dataset = BindingDBDataset(channel_idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size=modelArgs['batch_size'])
    er_dataset = BindingDBDataset(er_idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size=modelArgs['batch_size'])
    gpcr_dataset = BindingDBDataset(gpcr_idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size=modelArgs['batch_size'])
    kinase_dataset = BindingDBDataset(kinase_idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size=modelArgs['batch_size'])
    
    channel_loader = torch.utils.data.DataLoader(dataset = channel_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)
    er_loader = torch.utils.data.DataLoader(dataset = er_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)
    gpcr_loader = torch.utils.data.DataLoader(dataset = gpcr_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)
    kinase_loader = torch.utils.data.DataLoader(dataset = kinase_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)
    
    """Set a model."""
    model = Model(modelArgs)
    model.to(device)
    # PATH = './RESULT/'+'model--' + filesave+'.pth'
    PATH = '/NAS_Storage4/jaesuk/chem_esm_cont/src/RESULT/model--240821_fuse_ones.pth'
    model.load_state_dict(torch.load(PATH, map_location=device), strict=False)

    criterion = F.mse_loss
    tester = Tester(model, args.data_name)
    
    rmse, pearson = tester.test(kinase_loader)
    print('kinase')
    print('rmse\tpearson')
    result = [round(rmse,6),round(pearson,6)]
    print('\t '.join(map(str, result))) 

    rmse, pearson = tester.test(channel_loader)
    print('channel')
    print('rmse\tpearson')
    result = [round(rmse,6),round(pearson,6)]
    print('\t '.join(map(str, result))) 

    rmse, pearson = tester.test(er_loader)
    print('er')
    print('rmse\tpearson')
    result = [round(rmse,6),round(pearson,6)]
    print('\t '.join(map(str, result))) 

    rmse, pearson = tester.test(gpcr_loader)
    print('gpcr')
    print('rmse\tpearson')
    result = [round(rmse,6),round(pearson,6)]
    print('\t '.join(map(str, result))) 
else:
    print('wrong data name')
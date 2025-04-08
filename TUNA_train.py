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
parser.add_argument('-train', '--bool_train', help='training? or test?', type=bool, default=True)
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


print('modelArgs : ',modelArgs)

#setting wandb
if args.project_name!='non':
    import wandb

    wandb.init(project=args.project_name)

    wandb.run.name = filesave
    wandb.run.save()

    wandb.config.update(modelArgs)
else:
    None
    
class Trainer(object):
    def __init__(self, model, data_name):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),lr=modelArgs['lr'], weight_decay=modelArgs['weight_decay'])
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=modelArgs['T_max'], eta_min=modelArgs['lr']*0.1)
        self.data_name = data_name
        
    def train(self, dataloader):
        max_norm = 5
        self.model.train()
        N = len(dataloader)
        loss_total = 0
        
        for batch_idx, samples in enumerate(dataloader):
            
            samples[0].seq = samples[0].seq.view(-1,1000,1280)
            
            samples[0].encode_input = samples[0].encode_input.view(-1,102)
            samples[0].encoder_pad_mask = samples[0].encoder_pad_mask.view(-1,102)
            
            if self.data_name=='pdbbind':
                samples[0].pk = samples[0].pk.view(-1, 82) 
            elif self.data_name=='bindingdb':
                samples[0].pk = samples[0].pk.view(-1, 434)
            else:
                print('wrong data name')
            
            loss = self.model(samples)
            
                
            l1_reg = torch.tensor(0., requires_grad=True)

            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.linalg.norm(param, 1)

            
            loss = loss + 10e-4 * l1_reg
                
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()
            loss_total += loss.item() 
            
        self.scheduler.step()
            
        return loss_total


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
                
        pearson = CORR(T,Y)
        rmse = RMSE(T,Y)
        mae = MAE(T, Y)
        sd = SD(T, Y)
        ci = c_index(T, Y)
        return pearson, rmse, mae, sd, ci

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
    com_node, com_edge, com_padidx, com_token, com_pad, sequences, train_pockets, val_pockets = load_data(args.data_name, args.bool_cv, args.bool_train, args.fold)
    train_idx, val_idx = load_label(args.data_name, args.bool_cv, args.bool_train, args.fold)

    train_dataset = PDBbindDataset(idx=train_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=train_pockets, batch_size=modelArgs['batch_size'])
    val_dataset = PDBbindDataset(idx=val_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=val_pockets, batch_size=modelArgs['batch_size'])                         

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = modelArgs['batch_size'],shuffle = True, num_workers = 2, drop_last =True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = modelArgs['batch_size'], num_workers = 2, drop_last =True, collate_fn=collate_fn)


    model = Model(modelArgs)
    model.to(device)

    criterion = F.mse_loss
    trainer = Trainer(model, args.data_name)
    tester = Tester(model, args.data_name)

    """Output files."""
    setting = modelArgs['filesave']
    file_results = './RESULT/Results--' + setting + '.txt'
    file_model = './RESULT/model--' + setting+ '.pth'
    Results = ('Epoch\tTime(sec)\tLoss_train\tpearson_val\trmse_val\tmae_val\tsd_val\tci_val')
    with open(file_results, 'w') as f:
        f.write(Results + '\n')

    """Start training."""
    print('Training...')
    print(Results)
    start = time.process_time()

    best_pearson = 0
    pear_val = []

    for epoch in range(1,modelArgs['iteration']):
        if epoch % modelArgs['decay_interval'] == 0:
            trainer.optimizer.param_groups[0]['lr'] *= modelArgs['lr_decay']
        loss_train = trainer.train(train_loader)
        pearson_val, rmse_val, mae_val, sd_val, ci_val = tester.test(val_loader)

        end = time.process_time()
        timeforepoch = int(end - start)
        
        result = [epoch, timedelta(seconds=timeforepoch), round(loss_train,5), round(pearson_val,5),
                round(rmse_val,5), round(mae_val,5), round(sd_val,5), round(ci_val,5)]
        
        wandb.log({'loss_train': loss_train},step=epoch)
        wandb.log({'pearson_val': pearson_val},step=epoch)
        wandb.log({'rmse_val': rmse_val},step=epoch)
        wandb.log({'mae_val': mae_val},step=epoch)
        wandb.log({'sd_val': sd_val},step=epoch)
        wandb.log({'ci_val': ci_val},step=epoch)
        
        tester.save_rmse(result, file_results)
        
        if pearson_val > best_pearson:
            tester.save_model(model, file_model)
            
            best_pearson = pearson_val
            best_rmse = rmse_val
            best_mae = mae_val
            best_sd = sd_val
            best_ci = ci_val
            patience = 0
        else:
            patience+=1
        if patience == args.patience:
            print('patience early stop')
            print("Best Val Pearson : ",best_pearson)
            print("Best Val RMSE : ",best_rmse)
            print("Best Val MAE : ",best_mae)
            print("Best Val SD : ",best_sd)
            print("Best Val CI : ",best_ci)
            break

        print('\t '.join(map(str, result)))
        
elif args.data_name=='bindingdb':
    com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data = load_data(args.data_name, args.bool_cv, args.bool_train, args.fold)
    train_idx, val_idx, test_idx = load_label(args.data_name, args.bool_cv, args.bool_train, args.fold)
    
    train_dataset = BindingDBDataset(train_idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size=modelArgs['batch_size'])
    val_dataset = BindingDBDataset(val_idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size=modelArgs['batch_size'])
    test_dataset = BindingDBDataset(test_idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size=modelArgs['batch_size'])

    

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = modelArgs['batch_size'],shuffle = True, num_workers = 2, drop_last =True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = modelArgs['batch_size'], num_workers = 2, drop_last =True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = modelArgs['batch_size'], num_workers = 2, drop_last =True, collate_fn=collate_fn)
    
    model = Model(modelArgs)
    model.to(device)
    
    criterion = F.mse_loss
    trainer = Trainer(model, args.data_name)
    tester = Tester(model, args.data_name)

    """Output files."""
    setting = modelArgs['filesave']
    file_results = './RESULT/Results--' + setting + '.txt'
    file_model = './RESULT/model--' + setting+ '.pth'
    Results = ('Epoch\tTime(sec)\tLoss_train\tpearson_val\trmse_val\tmae_val\tsd_val\tci_val')
    with open(file_results, 'w') as f:
        f.write(Results + '\n')

    """Start training."""
    print('Training...')
    print(Results)
    start = time.process_time()

    best_pearson = 0
    pear_val = []
    
    for epoch in range(1,modelArgs['iteration']):
        if epoch % modelArgs['decay_interval'] == 0:
            trainer.optimizer.param_groups[0]['lr'] *= modelArgs['lr_decay']
        loss_train = trainer.train(train_loader)
        rmse_val, pearson_val = tester.test(val_loader)
        rmse_test, pearson_test = tester.test(test_loader)

        end = time.process_time()
        timeforepoch = int(end - start)
        
        result = [epoch, timedelta(seconds=timeforepoch), round(loss_train,5), round(rmse_val,5),
                round(pearson_val,5),round(rmse_test,5),round(pearson_test,5)]
        
        wandb.log({'loss_train': loss_train},step=epoch)
        wandb.log({'rmse_val': rmse_val},step=epoch)
        wandb.log({'rmse_test': rmse_test},step=epoch)
        wandb.log({'pearson_val': pearson_val},step=epoch)
        wandb.log({'pearson_test': pearson_test},step=epoch)
        
        tester.save_rmse(result, file_results)
        
        if pearson_val > best_pearson:
            tester.save_model(model, file_model)
            best_pearson = pearson_val
            best_test_rmse = rmse_test
            best_test_pearson = pearson_test
            patience = 0
        else:
            patience+=1
        if patience == args.patience:
            print('patience early stop')
            print("Best Val Pearson : ",best_pearson)
            print("Best Test RMSE : ", best_test_rmse)
            print("Best Test Pearson : ", best_test_pearson)
            break
        
        print('\t '.join(map(str, result)))
    
else:
    print('wrong data name')
    

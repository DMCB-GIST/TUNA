import time ###
start = time.time() ###

# Setting Device
import torch
import os
import wandb
import argparse 
from torch_geometric.data import Batch
import numpy as np
from scipy.stats import pearsonr
import torch.nn.functional as F
import torch.optim as optim
from datetime import timedelta
from utils import *
from dataset import CustomDataset
from models import Model
patience = 0



parser = argparse.ArgumentParser(description='training the model')   
parser.add_argument('-n', '--gpu', help='GPU number',type=str,required=True)   
parser.add_argument('-f', '--file', help='save file',type=str,required=True)   
parser.add_argument('-p', '--project_name', help='project_name',type=str,default='non')  
parser.add_argument('-hold', '--patience', help='early stop patience', type=int, default=10)
parser.add_argument('-train', '--bool_train', help='training? or test?', type=bool, default=False)
parser.add_argument('-cv', '--bool_cv', help='cross validation??', type=bool, default=False)
parser.add_argument('-fold', '--fold', help='dataset fold', type=int, default=0)

args = parser.parse_args()  

print(args)
filesave = args.file
GPU_NUM =  args.gpu # Number of GPU

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
print("CPU Random Seed :", torch.initial_seed())
print("CUDA Random Seed :", torch.cuda.initial_seed())

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


print('modelArgs : ',modelArgs)

#setting wandb
if args.bool_train:
    if args.project_name!='non':
        import wandb

        wandb.init(project=args.project_name)

        wandb.run.name = filesave
        wandb.run.save()

        wandb.config.update(modelArgs)
    else:
        None
else:
    None

"""Load preprocessed data."""
dir_input = '/NAS_Storage4/jaesuk/CAPLA/'
chemF_dir_input = '/NAS_Storage4/jaesuk/Chemformer/'

sequences = np.load(dir_input+'PDBbind_AASeqESM2.npy', allow_pickle= True)

if args.bool_cv:
    train_pockets = np.load(dir_input+f'pocket_data/Pk_train_esm_AA_5fold_{args.fold}_sd_42.npy', allow_pickle= True)
    val_pockets = np.load(dir_input+f'pocket_data/Pk_val_esm_AA_5fold_{args.fold}_sd_42.npy', allow_pickle= True)
    
else:
    train_pockets = np.load(dir_input+'pocket_data/Pk_train_esm_AA_sd42.npy', allow_pickle= True)
    val_pockets = np.load(dir_input+'pocket_data/Pk_val_esm_AA_sd42.npy', allow_pickle= True)
test195_pockets = np.load(dir_input+'pocket_data/Pk_195_esm_AA.npy', allow_pickle= True)
test262_pockets = np.load(dir_input+'pocket_data/Pk_262_esm_AA.npy', allow_pickle= True)
test290_pockets = np.load(dir_input+'pocket_data/Pk_290_esm_AA.npy', allow_pickle= True)
test95_pockets = np.load(dir_input+'pocket_data/Pk_95_esm_AA.npy', allow_pickle= True)
    
print("Diff Graph data load ...")
com_node = np.load(dir_input + 'Diffusion_Graph/PDBbind_ComGraph_node.npy', allow_pickle=True)
com_edge = np.load(dir_input + 'Diffusion_Graph/PDBbind_ComGraph_edge.npy', allow_pickle=True)
com_padidx = np.load(dir_input + 'Diffusion_Graph/PDBbind_ComGraph_padidx.npy', allow_pickle=True)

print("ChemF data load ...")
com_token = np.load(dir_input + "PDBbind_ComChemF_input.npy", allow_pickle=True)
com_pad = np.load(dir_input + "PDBbind_ComChemF_pad.npy", allow_pickle=True)

if args.bool_train:
    if args.bool_cv:
        fold_num = args.fold
        print("Cross Validation Fold :", fold_num)
        train_idx = load_pickle(dir_input + f'labels_train_5fold_{fold_num}_sd_42.pkl')
        val_idx = load_pickle(dir_input + f'labels_val_5fold_{fold_num}_sd_42.pkl')
        
        print('size of dataset_train : ',len(train_idx))
        print('size of dataset_dev : ',len(val_idx))
    else:
        train_idx = load_pickle(dir_input + 'labels_Training_data_idx_set_sd42.pkl')
        val_idx = load_pickle(dir_input + 'labels_Validation_data_idx_set_sd42.pkl')
        print('size of dataset_train : ',len(train_idx))
        print('size of dataset_dev : ',len(val_idx))

else:
    test2013_195_idx = load_pickle(dir_input+'labels_Test2013_195_idx_set.pkl')
    test2013_95_idx = load_pickle(dir_input+'labels_Test2013_95_idx_set.pkl')
    test2016_262_idx = load_pickle(dir_input+'labels_Test2016_262_idx_set.pkl')
    test2016_290_idx = load_pickle(dir_input+'labels_Test2016_290_idx_set.pkl')


    print('size of channel : ',len(test2013_195_idx))
    print('size of er : ',len(test2013_95_idx))
    print('size of gpcr : ',len(test2016_262_idx))
    print('size of kinase : ',len(test2016_290_idx))


# Setting Device


torch.autograd.set_detect_anomaly(True)

max_norm = 5

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),lr=modelArgs['lr'], weight_decay=modelArgs['weight_decay'])
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=modelArgs['T_max'], eta_min=modelArgs['lr']*0.1)
        
    def train(self, dataloader):
        self.model.train()
        N = len(dataloader)
        loss_total = 0
        
        for batch_idx, samples in enumerate(dataloader):
            
            samples[0].seq = samples[0].seq.view(-1,1000,1280)
            
            samples[0].encode_input = samples[0].encode_input.view(-1,102)
            samples[0].encoder_pad_mask = samples[0].encoder_pad_mask.view(-1,102)
            
            samples[0].pk = samples[0].pk.view(-1, 82) 
            
            
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
    def __init__(self, model):
        self.model = model

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
                samples[0].pk = samples[0].pk.view(-1, 82) 
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

"""Set a model."""

if args.bool_train:
    train_dataset = CustomDataset(idx=train_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=train_pockets, batch_size=modelArgs['batch_size'])
    val_dataset = CustomDataset(idx=val_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=val_pockets, batch_size=modelArgs['batch_size'])                         

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = modelArgs['batch_size'],shuffle = True, num_workers = 2, drop_last =True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = modelArgs['batch_size'], num_workers = 2, drop_last =True, collate_fn=collate_fn)


    model = Model(modelArgs, device=device)
    model.to(device)

    criterion = F.mse_loss
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    setting = modelArgs['filesave']
    file_AUCs = './RESULT/AUCs--' + setting + '.txt'
    file_model = './RESULT/model--' + setting+ '.pth'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tpearson_val\trmse_val\tmae_val\tsd_val\tci_val')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    import time
    from datetime import timedelta
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
        
        tester.save_rmse(result, file_AUCs)
        
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
        
else:
    
    test2013_195_dataset = CustomDataset(idx=test2013_195_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=test195_pockets, batch_size=modelArgs['batch_size'])
    test2016_290_dataset = CustomDataset(idx=test2016_290_idx, com_node=com_node, com_edge=com_edge, com_padidx=com_padidx, com_token=com_token, com_pad=com_pad, sequences=sequences, pk_data=test290_pockets, batch_size=modelArgs['batch_size'])

    test2013_195_loader = torch.utils.data.DataLoader(dataset = test2013_195_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)
    test2016_290_loader = torch.utils.data.DataLoader(dataset = test2016_290_dataset,batch_size = modelArgs['batch_size'],num_workers = 2, collate_fn=collate_fn)

    """Set a model."""
    model = Model(modelArgs, device=device)
    model.to(device)
    # PATH = './RESULT/'+'model--' + filesave+'.pth'
    PATH = '/NAS_Storage4/jaesuk/chem_esm_cont/src/RESULT/model--240821_fuse_ones.pth'
    model.load_state_dict(torch.load(PATH, map_location=device))

    criterion = F.mse_loss
    tester = Tester(model)

    pearson, rmse, mae, sd, ci = tester.test(test2013_195_loader)
    print('test2013_195')
    print('pearson\trmse\tmae\tsd\tci')
    result = [round(pearson,6),round(rmse,6),round(mae,6),round(sd,6),round(ci,6)]
    print('\t '.join(map(str, result))) 

    pearson, rmse, mae, sd, ci = tester.test(test2016_290_loader)
    print('test2016_290')
    print('pearson\trmse\tmae\tsd\tci')
    result = [round(pearson,6),round(rmse,6),round(mae,6),round(sd,6),round(ci,6)]
    print('\t '.join(map(str, result))) 
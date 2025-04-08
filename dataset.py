import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from utils import load_pickle
class PDBbindDataset(Dataset): 
    def __init__(self, idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size):
        self.idx = idx
        self.com_node = com_node
        self.com_edge = com_edge
        self.com_padidx = com_padidx
        self.com_token = com_token
        self.com_pad = com_pad
        self.sequences = sequences
        self.pk_data = pk_data
        self.batch_size = batch_size

    def __len__(self): 
        return len(self.idx)
 
    def __getitem__(self, i): 
        
        pro_idx = self.idx[i][0]
        com_idx = self.idx[i][1]
        
        node_feat = torch.FloatTensor(self.com_node.item().get(com_idx))
        edge_index = torch.LongTensor(self.com_edge.item().get(com_idx))
        pad_idx = torch.LongTensor(self.com_padidx.item().get(com_idx))
        
        encode_input = torch.LongTensor(self.com_token.item().get(com_idx))
        encoder_pad_mask = torch.BoolTensor(self.com_pad.item().get(com_idx))
        
        seq = torch.FloatTensor(self.sequences.item().get(pro_idx))
        
        pk = torch.LongTensor(self.pk_data)[i]
        
        data = Data(x=node_feat, edge_index=edge_index, seq=seq, pk=pk, encode_input=encode_input, encoder_pad_mask=encoder_pad_mask, pad_idx=pad_idx, batch=self.batch_size)
            
        x = data
        
        y = torch.FloatTensor([self.idx[i][2]])

        return x, y
    
class BindingDBDataset(Dataset): 
    def __init__(self, idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size):
        self.idx = idx
        self.com_node = com_node
        self.com_edge = com_edge
        self.com_padidx = com_padidx
        self.com_token = com_token
        self.com_pad = com_pad
        self.sequences = sequences
        self.pk_data = pk_data
        self.batch_size = batch_size

    def __len__(self): 
        return len(self.idx)
 
    def __getitem__(self, i): 
        
        pro_idx = self.idx[i][0]
        com_idx = self.idx[i][1]
        
        node_feat = torch.FloatTensor(self.com_node.item().get(com_idx))
        edge_index = torch.LongTensor(self.com_edge.item().get(com_idx))
        pad_idx = torch.LongTensor(self.com_padidx.item().get(com_idx))
        
        encode_input = torch.LongTensor(self.com_token.item().get(com_idx))
        encoder_pad_mask = torch.BoolTensor(self.com_pad.item().get(com_idx))

        seq = torch.FloatTensor(self.sequences.item().get(pro_idx))
        
        pk = torch.LongTensor(self.pk_data.item().get(pro_idx))
        
        data = Data(x=node_feat, edge_index=edge_index, seq=seq, pk=pk, encode_input=encode_input, encoder_pad_mask=encoder_pad_mask, pad_idx=pad_idx, batch=self.batch_size)
        
        x = data

        y = torch.FloatTensor([self.idx[i][2]])

        return x, y
    
def load_data(data_name, bool_cv, bool_train, fold_num=0):
    if data_name=="pdbbind":
        """Load preprocessed data."""
        dir_input = f'./data/{data_name}/'
        
        sequences = np.load(dir_input+'PDBbind_AASeqESM2.npy', allow_pickle= True)

        print("Diff Graph data load ...")
        com_node = np.load(dir_input + 'Diffusion_Graph/PDBbind_ComGraph_node.npy', allow_pickle=True)
        com_edge = np.load(dir_input + 'Diffusion_Graph/PDBbind_ComGraph_edge.npy', allow_pickle=True)
        com_padidx = np.load(dir_input + 'Diffusion_Graph/PDBbind_ComGraph_padidx.npy', allow_pickle=True)

        print("ChemF data load ...")
        com_token = np.load(dir_input + "PDBbind_ComChemF_input.npy", allow_pickle=True)
        com_pad = np.load(dir_input + "PDBbind_ComChemF_pad.npy", allow_pickle=True)
        
        print("pocket data load ...")
        if bool_train:
            
            if bool_cv:
                train_pockets = np.load(dir_input+f'pocket_data/Pk_train_esm_AA_5fold_{fold_num}.npy', allow_pickle= True)
                val_pockets = np.load(dir_input+f'pocket_data/Pk_val_esm_AA_5fold_{fold_num}.npy', allow_pickle= True)
                
            else:
                train_pockets = np.load(dir_input+'pocket_data/Pk_train_esm_AA.npy', allow_pickle= True)
                val_pockets = np.load(dir_input+'pocket_data/Pk_val_esm_AA.npy', allow_pickle= True)
            
            return com_node, com_edge, com_padidx, com_token, com_pad, sequences, train_pockets, val_pockets
        
        else:
            test195_pockets = np.load(dir_input+'pocket_data/Pk_195_esm_AA.npy', allow_pickle= True)
            test262_pockets = np.load(dir_input+'pocket_data/Pk_262_esm_AA.npy', allow_pickle= True)
            test290_pockets = np.load(dir_input+'pocket_data/Pk_290_esm_AA.npy', allow_pickle= True)
            test95_pockets = np.load(dir_input+'pocket_data/Pk_95_esm_AA.npy', allow_pickle= True)

            return com_node, com_edge, com_padidx, com_token, com_pad, sequences, test195_pockets, test262_pockets, test290_pockets, test95_pockets
    elif data_name=="bindingdb":
        """Load preprocessed data."""
        dir_input = f'./data/{data_name}/'
        sequences = np.load(dir_input+'F_AASeqESM2.npy', allow_pickle= True)
        
        pockets = np.load(dir_input+'pocket_data/F_PkAASeq_esm2.npy', allow_pickle= True)
        
        com_node = np.load(dir_input + 'Diffusion_Graph/F_ComGraph_node.npy', allow_pickle=True)
        com_edge = np.load(dir_input + 'Diffusion_Graph/F_ComGraph_edge.npy', allow_pickle=True)
        com_padidx = np.load(dir_input + 'Diffusion_Graph/F_ComGraph_padidx.npy', allow_pickle=True)

        com_token = np.load(dir_input + "F_ComChemF_input.npy", allow_pickle=True)
        com_pad = np.load(dir_input + "F_ComChemF_pad.npy", allow_pickle=True)
        
        return com_node, com_edge, com_padidx, com_token, com_pad, sequences, pockets
    else:
        print("wrong data name")

def load_label(data_name, bool_cv, bool_train, fold_num=0):
    
    if data_name=="pdbbind":
        dir_input = f'./data/{data_name}/'
        if bool_train:
            if bool_cv:
                print("Cross Validation Fold :", fold_num)
                train_idx = load_pickle(dir_input + f'labels_train_5fold_{fold_num}.pkl')
                val_idx = load_pickle(dir_input + f'labels_val_5fold_{fold_num}.pkl')
                
                print('size of dataset_train : ',len(train_idx))
                print('size of dataset_dev : ',len(val_idx))
            else:
                train_idx = load_pickle(dir_input + 'labels_Training_data_idx_set.pkl')
                val_idx = load_pickle(dir_input + 'labels_Validation_data_idx_set.pkl')
                print('size of dataset_train : ',len(train_idx))
                print('size of dataset_dev : ',len(val_idx))
                
            return train_idx, val_idx

        else:
            test2013_195_idx = load_pickle(dir_input+'labels_Test2013_195_idx_set.pkl')
            test2013_95_idx = load_pickle(dir_input+'labels_Test2013_95_idx_set.pkl')
            test2016_262_idx = load_pickle(dir_input+'labels_Test2016_262_idx_set.pkl')
            test2016_290_idx = load_pickle(dir_input+'labels_Test2016_290_idx_set.pkl')


            print('size of test 195 : ',len(test2013_195_idx))
            print('size of test 95 : ',len(test2013_95_idx))
            print('size of test 262 : ',len(test2016_262_idx))
            print('size of test 290 : ',len(test2016_290_idx))
            
            return test2013_195_idx, test2013_95_idx, test2016_262_idx, test2016_290_idx
    
    elif data_name=="bindingdb":
        dir_input = f'./data/{data_name}/'
        if bool_train:
            if bool_cv:
                from sklearn.model_selection import train_test_split
                print("Cross Validation Fold :", fold_num)
                train_idx = load_pickle(dir_input + f'labels_train_5fold_{fold_num}.pkl')
                test_idx = load_pickle(dir_input + f'labels_test_5fold_{fold_num}.pkl')
                
                train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
                
                print('size of dataset_train : ',len(train_idx))
                print('size of dataset_dev : ',len(val_idx))
                print('size of dataset_test : ',len(test_idx))
            else:
                train_idx = load_pickle(dir_input + 'labels_train_set.pkl')
                test_idx = load_pickle(dir_input + 'labels_test_set.pkl')
                
                train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
                
                print('size of dataset_train : ',len(train_idx))
                print('size of dataset_dev : ',len(val_idx))
                print('size of dataset_test : ',len(test_idx))
                
            return train_idx, val_idx, test_idx
        else:
            channel_idx = load_pickle(dir_input+'labels_channel_idx_set.pkl')
            er_idx = load_pickle(dir_input+'labels_ER_idx_set.pkl')
            gpcr_idx = load_pickle(dir_input+'labels_GPCR_idx_set.pkl')
            kinase_idx = load_pickle(dir_input+'labels_kinase_idx_set.pkl')


            # print('size of dataset_test : ',len(test_idx))
            print('size of channel : ',len(channel_idx))
            print('size of er : ',len(er_idx))
            print('size of gpcr : ',len(gpcr_idx))
            print('size of kinase : ',len(kinase_idx))
            
            return channel_idx, er_idx, gpcr_idx, kinase_idx
    else:
        print("wrong data name")
        
        
    
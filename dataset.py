import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class CustomDataset(Dataset): 
    def __init__(self,idx, com_node, com_edge, com_padidx, com_token, com_pad, sequences, pk_data, batch_size):
        self.idx = idx
        self.com_node = com_node
        self.com_edge = com_edge
        self.com_padidx = com_padidx
        self.com_token = com_token
        self.com_pad = com_pad
        self.sequences = sequences
        self.pk_data = pk_data
        self.batch_size = batch_size

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.idx)
 
    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
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
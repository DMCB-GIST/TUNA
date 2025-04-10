import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from  torch_geometric.nn import GCNConv
from torch import Tensor
from collections import OrderedDict
from torch_geometric.data import DataLoader
import pdb
from typing import Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)
    
class convLayerIndReLU(nn.Module):

    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.convLayerInd(in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, 
                      out_features=out_features, 
                      bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', 
                                               convLayerIndReLU(in_channels, 
                                                                out_channels, 
                                                                kernel_size=kernel_size, 
                                                                stride=stride, 
                                                                padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), 
                                convLayerIndReLU(out_channels, 
                                                 out_channels, 
                                                 kernel_size=kernel_size, 
                                                 stride=stride, 
                                                 padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):

        return self.inc(x).squeeze(-1)

class TargetRepresentation(nn.Module):
    def __init__(self, block_num, 
                 vocab_size, 
                 embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 
                                  embedding_num, 
                                  padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx+1, 
                         embedding_num, 
                         96, 
                         3)
            )

        self.linear = nn.Linear(block_num * 96, 96)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x

class NodeBN(_BatchNorm):

    def __init__(self, num_features, 
                 eps=1e-5, 
                 momentum=0.1, 
                 affine=True,
                 track_running_stats=True):
        super(NodeBN, self).__init__(
            num_features, 
            eps, momentum, 
            affine, 
            track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, 
            self.running_mean, 
            self.running_var, 
            self.weight, 
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, 
            self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)



class DGraphConvBn(nn.Module):
    def __init__(self, in_channels, 
                 hidden_channels, 
                 out_channels):
        super().__init__()
        self.convLayerIn = gnn.GCNConv(in_channels, hidden_channels)
        self.convLayerOut = gnn.GCNConv(hidden_channels, out_channels)
        self.norm = NodeBN(out_channels)

        self.diffusion_iterations = 8
        self.diffusion_weight = 0.1

    def forward(self, data):
        
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 데이터가 GPU에 있는지 확인
        if x.is_cuda:
            device = x.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = x.to(device)
            edge_index = edge_index.to(device)
            batch = batch.to(device)

        
        data.x = F.relu(self.norm(self.convLayerOut(self.convLayerIn(x, edge_index), edge_index)))

        adj_t = torch.sparse_coo_tensor(edge_index, torch.ones_like(edge_index[0]).float(), (x.size(0), x.size(0)), device=device)
        adj_t = adj_t.to_dense()
        adj_t = adj_t + torch.eye(x.size(0), device=device)
        for t in range(self.diffusion_iterations):
            adj_t = self.diffusion_weight * torch.matmul(adj_t, adj_t) + (1 - self.diffusion_weight) * torch.eye(x.size(0), device=device)
        adj_t = adj_t / adj_t.sum(dim=1, keepdim=True)

        laplacian = torch.eye(adj_t.size(0), device=device) - adj_t
        reg_term = torch.trace(torch.matmul(torch.transpose(x, 0, 1), laplacian))

        
        x = torch.matmul(adj_t, x) + reg_term

        return data
        


class DenseLayer(nn.Module):
    def __init__(self, 
                 num_input_features, 
                 growth_rate=32, 
                 bn_size=4):
        super().__init__()
        self.convLayerIn = DGraphConvBn(num_input_features, 
                                       16, 
                                       int(growth_rate * bn_size))
        self.convLayerOut = DGraphConvBn(int(growth_rate * bn_size), 
                                        32, 
                                        growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.convLayerIn(data)

        return data
    
    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.convLayerOut(data)

        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, 
                 num_input_features, 
                 growth_rate=32, 
                 bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, 
                               growth_rate, 
                               bn_size)
            self.add_module('layer%d' % (i + 1), layer)


    def forward(self, data):
        features = [data.x]

        
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, 
                 num_input_features, 
                 out_dim, 
                 growth_rate=32, 
                 block_config=(3, 3, 3, 3), 
                 bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('convn', DGraphConvBn(num_input_features, 16, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers,
                num_input_features, 
                growth_rate=growth_rate, 
                bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = DGraphConvBn(num_input_features, 32, num_input_features // 2)
            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):

        data = self.features(data)
        # x = gnn.global_mean_pool(data.x, data.batch)

        
        x = data.x
        x = self.classifer(x)

        return x


class Diffusion(nn.Module):
    def __init__(self, 
                 input_features=43,
                 filter_num=32, pad_mode=None):
        super().__init__()
        
        self.ligand_encoder = GraphDenseNet(num_input_features=input_features, 
                                            out_dim=filter_num, 
                                            block_config=[8, 8, 8], 
                                            bn_sizes=[2, 2, 2])
        self.pad_mode = pad_mode
        self.layernorm = Fp32LayerNorm(filter_num, eps=1e-12)
        
    def feat_pad(self, data, feat):
        idx = data.pad_idx.to(device)  # 예시 idx
        max_length=100
        
        # def apply_idx_to_tensor(temp_node_feat, idx, max_length=100):
        original_num_atoms, feature_size = feat.shape
        new_num_atoms = min(len(idx), max_length)  # 새로운 텐서의 길이는 max_length 이하로 설정
        if self.pad_mode!=None:
            if self.pad_mode=='zeros':
                pad_node_feat = torch.zeros((new_num_atoms, feature_size), device=device)  # 새 텐서 초기화
            elif self.pad_mode=='ones':
                feat = torch.sigmoid(feat)
                pad_node_feat = torch.zeros((new_num_atoms, feature_size), device=device)  # 새 텐서 초기화
            original_index = 0  # 원래 텐서의 인덱스
            for i in range(new_num_atoms):
                if idx[i] == 1:  # idx가 1인 경우
                    if original_index < original_num_atoms:
                        pad_node_feat[i] = feat[original_index]  # 기존 텐서의 원자 feature 삽입
                        original_index += 1  # 기존 텐서의 인덱스 증가
                
                elif idx[i] == 0:
                    if self.pad_mode=='zeros':
                        pad_node_feat[i] = torch.zeros((1, feature_size), device=device)
                        
                    elif self.pad_mode=='ones':
                        pad_node_feat[i] = torch.ones((1, feature_size), device=device)
            
                        
            
            return pad_node_feat
            
        else:
            # print("wrong pad mode is selected... or you do not use")
            
            if feat.size(0) > 100:
                feat = feat[:100]
            else:
                padding_size = 100 - feat.size(0)
                feat = F.pad(feat, (0, 0, 0, padding_size)).to(device)
            
            return feat

        


    def forward(self, data):
        
        ligand_x = self.ligand_encoder(data)

        # 각 리간드의 토큰 갯수를 고려하여 배치에서 분리
        batch_indices = data.batch.to(device)
        unique_indices = torch.unique(batch_indices)
        
        

        ligands_features = []
        for idx in unique_indices:
            ligand_features = ligand_x[batch_indices == idx]
            ligand_features = self.layernorm(ligand_features)
            
            ligand_features = self.feat_pad(data, feat=ligand_features)
            ligands_features.append(ligand_features)
        
        
        # 텐서 리스트를 하나의 텐서로 변환
        ligand_x = torch.stack(ligands_features)

        return ligand_x


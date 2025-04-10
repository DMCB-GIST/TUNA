import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
import copy
import math
import molbart.util as util_chemF
from molbart.decoder import DecodeSampler
import Diffusion
import esm

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        #        self.gelu = GELU()
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

# y = self.self_attention(y, kv, kv, attn_bias)
    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        #
        # Attention analyse
        #        csvwriter = csv.writer(open("attention.csv","a+",newline=""))

#         temp = x.cpu().numpy()
#         #        temp = temp.argmax(axis = 2)
#         temp = temp.mean(axis=2)
# #        print(temp.shape)
#         if temp.shape == (290,2,63):
#             np.save("attention.npy",temp)


        #        
#        np.save("")
        #        csvwriter.writerows(temp.tolist())
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x

# self.smi_attention_poc = EncoderLayer(128, 128, 0.1, 0.1, 2)  # 注意力机制
# self.tdpoc_attention_tdlig = EncoderLayer(32, 64, 0.1, 0.1, 1)

# smi_embed = self.smi_attention_poc(smi_embed, pkt_embed) #smi_emb : 150*128
# pkt_embed = self.smi_attention_poc(pkt_embed, smi_attention) #pkt_emb : 64*128

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        y = self.self_attention(y, kv, kv, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

class BiLSTM_(nn.Module):
    def __init__(self, emb_dim, lstm_hid_dim, dropout, dim, batch_size, device):
        """
        Initializes parameters suggested in paper

        args:
            emb_dim     : {int} embeddings dimension
            dropout     : {float}
            lstm_hid_dim: {int} hidden dimension for lstm
            r           : {int} representation vector
            batch_size  : {int} batch_size used for training
        """
        super(BiLSTM_, self).__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,2,batch_first=True,bidirectional=True,dropout=dropout) 
        self.linear = torch.nn.Linear(2*lstm_hid_dim,dim)
        self.hidden_state = self.init_hidden(lstm_hid_dim, batch_size)
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
    
    def init_hidden(self, lstm_hid_dim, batch_size):
        hidden = torch.zeros(4, batch_size, lstm_hid_dim, requires_grad=True).to(self.device)
        cell = torch.zeros(4, batch_size, lstm_hid_dim, requires_grad=True).to(self.device)
        return (hidden,cell)
    
    def forward(self, com: Tensor) -> Tensor:
        
        
        h0 = torch.zeros(4, com.size(0), 64).to(self.device)
        c0 = torch.zeros(4, com.size(0), 64).to(self.device)

        outputs, hidden_state = self.lstm(com,(h0, c0))    
        outputs = self.leakyrelu(self.linear(outputs)) # [batch, 1000, 10]
        # outputs = torch.mean(outputs, 1)   # avg emb
        return outputs

class BiLSTM(nn.Module):
    def __init__(self, emb_dim, pro_rep_dim, lstm_hid_dim, dropout, dim, batch_size, device):
        """
        Initializes parameters suggested in paper

        args:
            emb_dim     : {int} embeddings dimension
            dropout     : {float}
            lstm_hid_dim: {int} hidden dimension for lstm
            dim           : {int} representation vector
            batch_size  : {int} batch_size used for training
        """
        super(BiLSTM, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(29, emb_dim)
        self.lstm = torch.nn.LSTM(pro_rep_dim, lstm_hid_dim, 2, batch_first=True, bidirectional=True, dropout=dropout) 
        self.linear = torch.nn.Linear(2*lstm_hid_dim, dim)
        self.hidden_state = self.init_hidden(lstm_hid_dim, batch_size)
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(0.1)
    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
    
    def init_hidden(self, lstm_hid_dim, batch_szie):
        hidden = torch.zeros(4, batch_szie, lstm_hid_dim, requires_grad=True).to(self.device)
        cell = torch.zeros(4, batch_szie, lstm_hid_dim, requires_grad=True).to(self.device)
        return (hidden,cell)
    
    def forward(self, pro: Tensor) -> Tensor:
        # seq_embed = self.embeddings(pro) 
        h0 = torch.zeros(4, pro.size(0), 64).to(self.device)
        c0 = torch.zeros(4, pro.size(0), 64).to(self.device)

        outputs, hidden_state = self.lstm(pro,(h0, c0))    
        outputs = self.leakyrelu(self.linear(outputs)) # [batch, 1000, 10]
        # outputs = torch.mean(outputs, 1)   # avg emb
        return outputs

    
class AttentionLayer(nn.Module):
    def __init__(self, q_fc, kv_fc, emb):
        super(AttentionLayer, self).__init__()
        self.q_fc = copy.deepcopy(q_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(kv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(kv_fc) # (d_embed, d_model)
        self.layerNorm1 = nn.LayerNorm(emb)
        self.layerNorm2 = nn.LayerNorm(emb)
        self.dropout = nn.Dropout(p=0.1)
        self.linear_net = nn.Sequential(
            nn.Linear(emb, 64),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(64, emb)
        )

    def calculate_attention(self, query, key, value, mask):
        # query, key, value: (n_batch, seq_len, d_k)
        # mask: (n_batch, seq_len, seq_len)        
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        self.score = attention_score
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)
        self.score = attention_prob
        out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
        return out
    
    def forward(self, query, key, value, mask=None):
        fc_query = self.q_fc(query)
        fc_key = self.k_fc(key)
        fc_value = self.v_fc(value)
        out = self.calculate_attention(fc_query, fc_key, fc_value, mask) # (n_batch, seq_len, d_k)
        out = self.layerNorm1(query + self.dropout(out))
        
        linear_out = self.linear_net(out)
        out = out + self.dropout(linear_out)
        out = self.layerNorm2(out)

        return out
    
def positional_encoding(seq_len, d_model):
        PE = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                PE[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                if i + 1 < d_model:
                    PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        return PE
    
class SequenceViT(nn.Module):
    def __init__(self, emb_dim, num_heads=5, num_layers=1, seq_len=1000):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.position_embed = nn.Parameter(positional_encoding(seq_len, emb_dim))

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads), num_layers=num_layers)
        
        
    def forward(self, x):
        
        x = x + self.position_embed[:x.size(1), :]  # Add positional embeddings.
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + N, E)

        x = self.transformer(x)  # Apply the transformer.
        
        return x
    
from typing import Any

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
    
    
####### Chemformer Load ##########
class chemF_args():
    model_path='./molbart/models/last.ckpt'
    products_path='bindingdb_embedding.pickle'
    vocab_path=util_chemF.DEFAULT_VOCAB_PATH
    chem_token_start_idx=util_chemF.DEFAULT_CHEM_TOKEN_START
    num_beams=10

class chemF:
    DEFAULT_NUM_BEAMS = 10

    def __init__(self, device):
        
        self.device = device
        self.chemF_args = chemF_args()
        self.model = self.model_loader()
        
    def model_loader(self): 
        # print("Building tokeniser...")
        tokeniser = util_chemF.load_tokeniser(self.chemF_args.vocab_path, self.chemF_args.chem_token_start_idx)
        # print("Finished tokeniser.")
        
        sampler = DecodeSampler(tokeniser, util_chemF.DEFAULT_MAX_SEQ_LEN)
        pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

        # print("Loading model...")
        model = util_chemF.load_bart_train(self.chemF_args, sampler)
        model = model.to(self.device)
        model.num_beams = self.chemF_args.num_beams
        sampler.max_seq_len = model.max_seq_len
        # print("Finished model.")

        return model

    def load_enc(self):
        enc = self.model.encode
        
        return enc
    
    
    
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.device = args['device']
        # chemF 인스턴스 생성
        chemF_instance = chemF(device=self.device)
        
        self.chemF_enc = chemF_instance.load_enc()
        
        self.diffusion = Diffusion.Diffusion(input_features=43, filter_num=args['emb_dim'], pad_mode='ones').to(self.device)
        
        # self.c_lstm = BiLSTM_(emb_dim=args['emb_dim'], lstm_hid_dim=args['lstm_hid_dim'], dropout=args['dropout'], dim=args['r'], batch_size=args['batch_size'], device=self.device)
        self.p_lstm = BiLSTM(emb_dim=args['emb_dim'], pro_rep_dim = args['pro_rep_dim'], lstm_hid_dim=args['lstm_hid_dim'], dropout=args['dropout'], dim=args['r'], batch_size=args['batch_size'], device=self.device)
        self.pk_lstm = BiLSTM_(emb_dim=args['emb_dim'], lstm_hid_dim=args['lstm_hid_dim'], dropout=args['dropout'], dim=args['r'], batch_size=args['batch_size'], device=self.device)
        
        self.pk_esm2, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.pk_esm2.load_state_dict(torch.load("/NAS_Storage4/jaesuk/PDBbind/esm2_best_model.pt", map_location=self.device))
        self.num_layers = 6
        self.pk_esm2.eval()
        for param in self.pk_esm2.parameters():
            param.requires_grad = False
            
        self.attention_com = AttentionLayer(q_fc = nn.Linear(args['r'], args['r']),kv_fc = nn.Linear(args['r'], args['r']), emb = args['r'])
        self.attention_pro = AttentionLayer(q_fc = nn.Linear(args['r'], args['r']),kv_fc = nn.Linear(args['r'], args['r']), emb = args['r'])
        self.pk_to_hidden = nn.Linear(args['pk_rep_dim'], args['emb_dim'])
        self.smi_attention_poc = EncoderLayer(args['emb_dim'], args['emb_dim'], 0.1, 0.1, 2)
        self.pro_enc = SequenceViT(emb_dim=args['r'], num_heads=1, num_layers=1)
        self.linear_com = nn.Linear(args['emb_dim'], args['r'])
        self.layernorm768 = Fp32LayerNorm(args['r'], eps=1e-12)
        self.layernorm512 = Fp32LayerNorm(args['emb_dim'], eps=1e-12)
        
        self.Linear_interaction = nn.Linear(3*args['r'], 1)
        
    def fuse_com(self, seq_com, graph_com):
        
        fused_com = seq_com # fused_com (102, 512)
        middle_seq_com = seq_com[:, 1:-1, :] # middle_seq_com (100, 512)
        product = graph_com.clone() * middle_seq_com.clone()
        
        fused_com[:, 1:-1, :] = product
        fused_com = self.linear_com(fused_com)
        fused_com = self.layernorm768(fused_com)
        return fused_com
    
    def chemF(self, data):
        com_enc = data.encode_input
        com_pad = data.encoder_pad_mask
        
        # chemformer apply
        encode_input = {
                    "encoder_input": com_enc.T,
                    "encoder_pad_mask": com_pad.T}
        
        com_vec = self.chemF_enc(encode_input)
        com_vec = com_vec.permute(1, 0, 2)
        com_vec = self.layernorm512(com_vec)
        
        
        return com_vec
    
    def pk_feat(self, data):
        pk_vec = data.pk
        
        pk_vec = self.pk_esm2(pk_vec, repr_layers=[self.num_layers])["representations"][self.num_layers]
        
        pk_vec = self.pk_to_hidden(pk_vec)
        pk_vec = self.layernorm512(pk_vec)
        
        return pk_vec
        
    def fuse_com_pk(self, com_vec, pk_vec):
    
        com_attention = com_vec
        
        com_vec = self.smi_attention_poc(com_vec, pk_vec)
        pk_vec = self.smi_attention_poc(pk_vec, com_attention)
        
        pk_vec = self.pk_lstm(pk_vec)
        
        com_vec = self.layernorm512(com_vec)
        pk_vec = self.layernorm768(pk_vec)
        
        return com_vec, pk_vec
        
        
    def pro_feat(self, data):
        pro_vec = data.seq
        
        pro_vec = self.p_lstm(pro_vec) # [batch, 1000, 10]
        
        pro_vec = self.pro_enc(pro_vec) 
        
        pro_vec = pro_vec[:,1:,:]
        pro_vec = self.layernorm768(pro_vec)
        
        return pro_vec
        
    def fuse_com_pro(self, fused_com_vec, pro_vec):
        temp_com_vec = fused_com_vec
        fused_com_vec = self.attention_com(fused_com_vec, pro_vec, pro_vec)
        pro_vec = self.attention_pro(pro_vec, temp_com_vec, temp_com_vec)
        
        fused_com_vec = self.layernorm768(fused_com_vec)
        pro_vec = self.layernorm768(pro_vec)
        
        return fused_com_vec, pro_vec
        
        
    def classifier(self, fused_com_vec, pro_vec, pk_vec):
        fused_com_vec = torch.mean(fused_com_vec, 1) # avg emb
        pro_vec = torch.mean(pro_vec, 1) # avg emb
        
        pk_vec = torch.mean(pk_vec, 1) # avg emb
        cat_vector = torch.cat((pro_vec, fused_com_vec, pk_vec), 1) # batch, 1000+432+100, 10
            
        
        
        outputs = self.Linear_interaction(cat_vector)
        
        return cat_vector, outputs
    
    def forward(self, inputs):
        
        
        data = inputs.to(self.device)
        # chemformer
        com_vec = self.chemF(data)

        # diffusion
        d_com_vec = self.diffusion(data)
        
        pk_vec = self.pk_feat(data)
        com_vec, pk_vec = self.fuse_com_pk(com_vec, pk_vec)
            


        pro_vec = self.pro_feat(data)

        fused_com_vec = self.fuse_com(com_vec, d_com_vec)
        
        fused_com_vec, pro_vec = self.fuse_com_pro(fused_com_vec, pro_vec)

        cat_vector, outputs = self.classifier(fused_com_vec, pro_vec, pk_vec)
            
        
        return cat_vector, outputs, pro_vec, com_vec

    def __call__(self, samples, train=True):
        inputs, y = samples
        true_aff = y
        true_aff = torch.squeeze(true_aff.to(self.device),0)
        cat_vector, pre_aff, pro_vec, com_vec = self.forward(inputs)

        mse = nn.MSELoss()

        if train:
            loss = mse(pre_aff,true_aff)
            
            return loss
        else:
            loss = mse(pre_aff,true_aff)
            true_aff = true_aff.to('cpu').data.numpy()
            pre_aff = pre_aff.to('cpu').data.numpy()
            loss = loss.to('cpu').data.numpy()  

            return true_aff, pre_aff, loss

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import hparams as hp
import sys
from torch.distributions import uniform
import numpy as np

class GST(nn.Module):

    def __init__(self):

        super().__init__()
        self.encoder = ReferenceEncoder()
        self.stl = STL()

    def forward(self, inputs, gst_index):
        
        if inputs is not None:
            print("USING REF")
            enc_out = self.encoder(inputs)
            style_embed = self.stl(enc_out, None)
        else:
            print("USING WEIGHTS")
            style_embed = self.stl(inputs, gst_index)
        
            

        return style_embed


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self):

        super().__init__()
        K = len(hp.ref_enc_filters)
#        print("K:", K)
        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.num_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.tts_embed_dims // 2,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hp.num_mels)  # [N, 1, Ty, n_mels]
        print("ref encoder out1:", out.size(), "should be [N, 1, Ty, n_mels]")
       
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]
        
        print("ref encoder out2:", out.size(), "should be [N, 128, Ty//2^K, n_mels//2^K]")

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        
        print("ref encoder out3:", out.size(), "should be [N, Ty//2^K, 128, n_mels//2^K]")

        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        print("ref encoder out4:", out.size(), "should be [N, Ty//2^K, 128*n_mels//2^K]")


        memory, out = self.gru(out)  # out --- [1, N, E//2]
        print("ref encoder out:", out.size(), "should be [1, N, E//2]")

        print("ref encoder out :", out.squeeze(0).size())

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.tts_embed_dims // hp.num_heads))
        d_q = hp.tts_embed_dims // 2
        d_k = hp.tts_embed_dims // hp.num_heads
        self.num_tokens = hp.token_num
        
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        print(d_q)
        print(d_k)
        self.allkeys = nn.Linear(in_features=d_k, out_features=hp.tts_embed_dims, bias=False)

        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hp.tts_embed_dims, num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs, gst_index):
        
        if inputs is not None:
            print("USING REF STL")

            print("Entering STL wuh inputs:", inputs.size() )
            N = inputs.size(0) #batch size
            query = inputs.unsqueeze(1)  # [N, 1, E//2]
            print("STL  query", query.size(), "should be [N, 1, E//2]")

            keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        
            print("STL keys", keys.size(), "should be [N, token_num, E // num_heads]")
        
            style_embed = self.attention(query, keys)
            print("final style embedding:", style_embed.size(), "should be [N,1,256]")
        else:

            print("Generating style using weights")
            N = 1
            keys = torch.tanh(self.embed.expand(N, -1, -1))
            #keys = self.allkeys(keys)
            key = keys.view(10,32) 
            labels = torch.arange(10)
            labels = labels.reshape(10, 1)
            num_classes = self.num_tokens
            one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes)).float()
           
            x = torch.Tensor(8,10)
            y = torch.rand(1,10)
            
            x = torch.zeros([8,10])
            print("TOKEN", gst_index) 
            y1 = torch.Tensor([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.8])
            y2 = torch.Tensor([0.1, 0.2, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0,0.8])
            y3 = torch.Tensor([0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.4, 0.0,0.0])

            
            x[0] = y3
            #x[1] = y2
            #x[2] = y3
            weights = x.cuda()
            #weights = torch.add(x,y)
            #weights = y.unsqueeze(0).repeat(1,8,1).view(8,10).cuda()
            #print("weights - before", weights)
            #distribution = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
            #weights = distribution.sample(torch.Size([8,10]))
            #weights = F.softmax(x, dim = -1).cuda()
            #weights = weights.view(8,10).cuda()
            print("weights", weights)
            #weight = np.transpose(one_hot_target).expand(N, -1, -1).cuda()
            
 
            print("keys 1", keys.size())
            print("weights", weights.size())
             
            #keys = keys.transpose(1, 2)
            #print("keys transpose", keys.size())

                       
                      
            
            out = torch.matmul( weights, keys)
            out = out.view(1,1,256)
            style_embed = out
            print(out.shape)
            #print(out)
            

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)


        print("multihead query:", querys.size(), "should be [N, T_q, num_units]")
        #print("multihead query length",querys[0][0])
        print("multihead keys:", keys.size(), "should be [N, T_k, num_units]")
       # print("multihead values:", values.size())
        
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        
        
        
        #print("multihead2 query:", querys.size())
        #print("multihead2 keys:", keys.size())
        #print("multihead2 values:", values.size())
        # score = softmax(QK^T / (d_k ** 0.5))
        
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        
        print("scores", scores.size(), "should be [h, N, T_q, T_k]")

        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        
        
        print("out 1", out.size(), "should be [h, N, T_q, num_units/h]")
        
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]


        print("out 2", out.size(), "should be [N, T_q, num_units]")

        return out


class MultiHeadAttention2(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 # dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        # if query_dim != key_dim:
        #     raise ValueError("query_dim and key_dim must be the same")
        # if num_units % h != 0:
        #     raise ValueError("num_units must be dividable by h")
        # if query_dim != num_units:
        #     raise ValueError("to employ residual connection, the number of "
        #                      "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        # self._key_dim = torch.tensor(
        #     data=[key_dim], requires_grad=True, dtype=torch.float32)
        self._key_dim = key_dim
        # self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        # self.bn = nn.BatchNorm1d(num_units)

    def forward(self, query, keys):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / (self._key_dim ** 0.5)
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            mask = torch.ones(diag_mat.size()) * (-2**32 + 1)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat - 1).abs())
        # put it to softmax
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        # attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)

        return attention

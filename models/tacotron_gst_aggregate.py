import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


import torch.nn.init as init
import hparams as hp
from torch.distributions import uniform




class GST(nn.Module):

    def __init__(self):

        super().__init__()
        self.ref_encoder = ReferenceEncoder()
   #     self.stl = STL()
        
    
    
        
    def forward(self, inputs, gst_index):
        
        if inputs is not None:
            print("USING REF")
            enc_out = self.ref_encoder(inputs)      
            style_embed = enc_out
            print(style_embed)


           # style_embed = style_embed.unsqueeze(1)
            #style_embed = self.stl(style_embed, None)
            #print(style_embed)
        else:
            print("USING WEIGHTS")
            style_embed = self.stl(inputs, gst_index)
        
            
        #print(style_embed)
        return style_embed

class BatchNormConv2D(nn.Module) :
    def __init__(self, in_channels, out_channels,  relu=True) :
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = relu

    def forward(self, x) :
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty, n_mels]  ref mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self):

        super().__init__()
        
        ref_enc_filters = hp.ref_enc_filters
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        
        self.convs = nn.ModuleList()
        for i in range(K):
            conv = BatchNormConv2D(in_channels=filters[i],out_channels=filters[i + 1] )
            self.convs.append(conv)
        
        
        self.gru = nn.GRU(256,128,batch_first=True)
        self.project = nn.Linear(128,256)


    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hp.num_mels)  # [N, 1, Ty, n_mels]
        print("ref encoder out1:", out.size(), "should be [N, 1, Ty, n_mels]")
       
        for conv in self.convs:
            out = conv(out)  # [N, 128, Ty//2^K, n_mels//2^K]
        


        print("ref encoder out2:", out.size(), "should be [N, 128, Ty//2^K, n_mels//2^K]")
        
        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        
        print("ref encoder out3:", out.size(), "should be [N, Ty//2^K, 128, n_mels//2^K]")

        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        print("ref encoder out4:", out.size(), "should be [N, Ty//2^K, 128*n_mels//2^K]")


        out, mem = self.gru(out)  # out --- [1, N, E//2]
        print("ref encoder encoder GRU hidden:", mem.size(), "should be [1, N, E//2]")
        print("ref encoder encoder GRU output:", out.size(), "should be [N, Ty//2^K, E//2]")
        
        print(out.size())
        
       # out =  out[:,-1,:]
        
        projected = self.project(out)
        print("ref encoder projected output:", projected.size(), "should be [N, E]")
        
        final_out = torch.tanh(projected)

        print("final output:", final_out.size(), "should be [N, E]")

        
        return final_out

    


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.tts_embed_dims // hp.num_heads))
        d_q = hp.tts_embed_dims 
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
            #query = inputs.unsqueeze(1)  # [N, 1, E//2]
            query = inputs
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
        #self.proj = nn.Linear(256,128, bias = False)
        
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

        self.out = nn.Linear(self.num_units, self.num_units, bias= False)
    def forward(self, query, key):
        #uery = self.proj(query)
        #print("here",query.size())
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)


        #print("multihead query:", querys, "should be [N, T_q, num_units]")
        #print("multihead query length",querys[0][0])
        #print("multihead keys:", keys, "should be [N,token_num , num_units]")
       # print("multihead values:", values.size())
         
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, token_num, _units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, token_num, num_units/h]

        
        
        querys = querys.transpose(0,1)
        keys = keys.transpose(0,1)
        values = values.transpose(0,1)
        print("multihead2 query:", querys.size())
        print("multihead2 keys:", keys.size())
        print("multihead2 values:", values.size())
        # score = softmax(QK^T / (d_k ** 0.5))
        print("transpose the keys:", keys.transpose(-2, -1).size()) 
        scores = torch.matmul(querys, keys.transpose(-2, -1))  # [ N,h, T_q, T_k]
        
        print("scores", scores.size(), "should be [N, h, T_q, token_num]")
        print("key_dim",self.key_dim)
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=-1)
        print("scores after softmax",scores.size(), "should be [N, h, T_q, token_num]")

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
       
    
        print("output",out.size(), "should be [N, h, T_q, num_units/h]")
        out = out.transpose(0,1)
        #concat = scores.transpose(1,2).contiguous().view(32,-1, self.key_dim)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        
        
        #out = self.out(out)
        print("out 2", out.size(), "should be [N, T_q, num_units]")
      #  print(out)
       
        return out
class HighwayNetwork(nn.Module) :
    def __init__(self, size) :
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)
        
    def forward(self, x) :
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class Encoder(nn.Module) : 
    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout) :
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels, 
                         proj_channels=[cbhg_channels, cbhg_channels], 
                         num_highways=num_highways)
        
    def forward(self, x) :
        x = self.embedding(x)
  
        x = self.pre_net(x)
        print("text encoder size after prenet", x.size())

        x.transpose_(1, 2)

        print("input to the CBHG" ,x.size())

        x, mem  = self.cbhg(x)
        
        return x, mem


class BatchNormConv(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel, relu=True) :
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu
        
    def forward(self, x) :
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)
    
    
class CBHG(nn.Module) :
    def __init__(self, K, in_channels, channels, proj_channels, num_highways) :
        super().__init__()
        
        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels :
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
                
        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)
        
        # Fix the highway input if necessary
        if proj_channels[-1] != channels :
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else :
            self.highway_mismatch = False
        
        self.highways = nn.ModuleList()
        for i in range(num_highways) :
            hn = HighwayNetwork(channels)
            self.highways.append(hn)
        
        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
    
    def forward(self, x) :

        # Save these for later
        residual = x
        seq_len = x.size(-1)
        conv_bank = []
        
        # Convolution Bank
        for conv in self.conv1d_bank :
            c = conv(x) # Convolution
            conv_bank.append(c[:, :, :seq_len])
        
        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)
        
        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len] 
        
        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)
        
        # Residual Connect
        x = x + residual
        
        # Through the highways
        x = x.transpose(1, 2)
        if self.highway_mismatch is True :
            x = self.pre_highway(x)
        for h in self.highways : x = h(x)

        # And then the RNN
        x, mem  = self.rnn(x)
        
        
        
        
        return x, mem


class PreNet(nn.Module) :
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5) :
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout
        
    def forward(self, x) :
        x = self.fc1(x)
        x = F.relu(x)
        #print("in pre-net", x)
        x = F.dropout(x, self.p, training=self.training)
        #print("in pre-net after dropout", x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x
    
    
class Attention(nn.Module) :
    def __init__(self, attn_dims) :
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        
        self.v = nn.Linear(attn_dims, 1, bias=False)
        
    def forward(self, encoder_seq_proj, query, t) :

        
        # Transform the query vector
        query_proj = self.W(query.unsqueeze(1))
        
        print(encoder_seq_proj.size())
        print(query_proj.size())
        
        # Compute the scores 
        #u= self.v(torch.tanh(torch.cat([encoder_seq_proj, query_proj],dim = 2)))
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))

        u = u.squeeze(-1)

        scores = F.softmax(u, dim=-1).unsqueeze(1)
        return scores


class LSA(nn.Module):
    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(2, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=False)
        self.L = nn.Linear(filters, attn_dim, bias=True)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None

    def init_attention(self, encoder_seq_proj) :
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t).cuda()
        self.attention = torch.zeros(b, t).cuda()

    def forward(self, encoder_seq_proj, query, t):

        if t == 0 : self.init_attention(encoder_seq_proj)

        processed_query = self.W(query).unsqueeze(1)

        location = torch.cat([self.cumulative.unsqueeze(1), self.attention.unsqueeze(1)], dim=1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))

        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)

        # Smooth Attention
        scores = torch.sigmoid(u) / torch.sigmoid(u).sum(dim=1, keepdim=True)
        # scores = F.softmax(u, dim=1)
        self.attention = scores
        self.cumulative += self.attention

        return scores.unsqueeze(-1).transpose(1, 2)


class Decoder(nn.Module) :
    def __init__(self, n_mels, decoder_dims, lstm_dims) :
        super().__init__()
        self.max_r = 20
        self.r = None
        self.generating = False
        self.n_mels = n_mels
        self.decoder_prenet = PreNet(n_mels)
        self.text_attn_net = Attention(decoder_dims)
        self.style_attn_net = Attention(decoder_dims)
        self.attn_net = Attention(decoder_dims)

        self.text_attn_rnn = nn.GRUCell(decoder_dims + decoder_dims//2,decoder_dims)
        self.style_attn_rnn = nn.GRUCell(decoder_dims + decoder_dims//2,decoder_dims)
        self.attn_rnn = nn.GRUCell(decoder_dims + decoder_dims//2,decoder_dims)

        
        self.rnn_input = nn.Linear(2 * decoder_dims, lstm_dims)        
        self.cv_project = nn.Linear(2 * decoder_dims, decoder_dims)

        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)
        
    def zoneout(self, prev, current, p=0.1) :
        mask = torch.zeros(prev.size()).bernoulli_(p).cuda()
        return prev * mask + current * (1 - mask)
    
    def forward(self, encoder_seq, style_embed, prenet_in, hidden_states, cell_states, context_vectors, t):
        
        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)
        T_x = encoder_seq.size(1)
        
        # Unpack the hidden and cell states
        text_attn_hidden, style_attn_hidden, attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states
        
        # Unpack context vectors
        text_context_vec, style_context_vec, context_vec = context_vectors

        # PreNet for the Attention RNN
       
        #print("decoder prenet", prenet_in.size())
        prenet_out = self.decoder_prenet(prenet_in)
        
        print("Decoder prenet output size", prenet_out.size())
        


        # Compute the Attention RNN hidden state
        text_attn_rnn_in = torch.cat([text_context_vec, prenet_out], dim=-1)  
        style_attn_rnn_in = torch.cat([style_context_vec, prenet_out], dim=-1)
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)


        print("Text Attention input", text_attn_rnn_in.squeeze(1).size())
        print("Style Attention input", style_attn_rnn_in.squeeze(1).size())
        #print("Attention input", attn_rnn_in.squeeze(1).size())

        
        # Text Attention
        text_attn_hidden = self.text_attn_rnn(text_attn_rnn_in.squeeze(1), text_attn_hidden)
        
        # Compute the text attention weights 
        text_scores = self.text_attn_net(encoder_seq, text_attn_hidden, t)
        
        print(text_scores.size())
        print(encoder_seq.size())

        # Dot product to create the text context vector 
        text_context_vec = text_scores @ encoder_seq

        text_context_vec = text_context_vec.squeeze(1)
        
        print(" Text Context Vector", text_context_vec.size())
        
         # Style Attention
        style_attn_hidden = self.style_attn_rnn(style_attn_rnn_in.squeeze(1), style_attn_hidden)

        # Compute the text attention weights 
        style_scores = self.style_attn_net(style_embed, style_attn_hidden, t)

        # Dot product to create the text context vector 
        style_context_vec = style_scores @ style_embed
        
        style_context_vec = style_context_vec.squeeze(1)
        
        print("Style Context Vector", style_context_vec.size())


        # Add context vectors
        #context_vec = torch.cat([text_context_vec, style_context_vec], dim = -1)
        #context_vec = style_context_vec + text_context_vec
        context_vec = text_context_vec

        #context_vec = self.cv_project(context_vec)

        print("Attention RNN input", attn_rnn_in.size())
        print("Attention RNN", attn_hidden.size())

        scores = (text_scores ,style_scores)

        print("Combined Context Vector and projected", context_vec.size())
        
        
        # Controller 1
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)
        print("Controller Attention RNN final hidden", attn_hidden.size())
        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1)
        print("concat full context and attention", x.size())
        x = self.rnn_input(x)
        print("Input to LSTM", x.size())

        # Controller 2

        #x1 = torch.cat([text_context_vec, text_attn_hidden], dim=1)
        #x2 = torch.cat([style_context_vec, style_attn_hidden], dim=1)

        #print("Controller Attention RNN final hidden", x1.size(), x2.size())

        #y = x1 + x2

        #print("Sum weights", y.size())

        #final_x = self.rnn_input(y)


        #print("rnn1", final_x.size())

        


        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if not self.generating :
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else :
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden



        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if not self.generating :
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else :
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden


        #x, hidden = self.res_rnn1(attn_project)
        #gru_out = x + attn_project
        
        #x2, hidden2 = self.res_rnn2(gru_out)
        #gru_out2 = x2 + gru_out


        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (text_attn_hidden, style_attn_hidden, attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)
        context_vectors = (text_context_vec, style_context_vec, context_vec)

        
        
        print("mel project", mels.size())
        
        
        return mels, scores, hidden_states, cell_states, context_vectors
    
    
class Tacotron(nn.Module) :
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout) :
        super().__init__()
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.decoder_dims = decoder_dims
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims, 
                               encoder_K, num_highways, dropout)
        self.encoder_proj = nn.Linear(decoder_dims, decoder_dims, bias=False)
        

        self.decoder = Decoder(n_mels, decoder_dims, lstm_dims)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims, [256, 80], num_highways)
        self.post_proj = nn.Linear(postnet_dims * 2, fft_bins, bias=False)

        self.init_model()
        self.num_params()
        self.gst = GST()

        # Unfortunately I have to put these settings into params in order to save
        # if anyone knows a better way of doing this please open an issue in the repo
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.r = nn.Parameter(torch.tensor(0).long(), requires_grad=False)

    def set_r(self, r) :
        self.r.data = torch.tensor(r)
        self.decoder.r = r

    def get_r(self) :
        return self.r.item()

    def forward(self, x, m, generate_gta=False) :
        print("Forward pass")
        self.step += 1

        if generate_gta :
            self.encoder.eval()
            self.postnet.eval()
            self.decoder.generating = True
        else :
            self.encoder.train()
            self.postnet.train()
            self.gst.train()
            self.decoder.generating = False
            self.training = True

        batch_size, n_mels, steps  = m.size()
         
        # Initialise all hidden states and pack into tuple
        text_attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()
        style_attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()
        attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()

        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        hidden_states = (text_attn_hidden, style_attn_hidden, attn_hidden, rnn1_hidden, rnn2_hidden)
        
        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        cell_states = (rnn1_cell, rnn2_cell)
        
        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels).cuda()
        
        # Need an initial context vector
        context_vec_text = torch.zeros(batch_size, self.decoder_dims).cuda()
        context_vec_style = torch.zeros(batch_size, self.decoder_dims).cuda()
        context_vec = torch.zeros(batch_size,  self.decoder_dims).cuda()
        
        context_vectors = (context_vec_text, context_vec_style, context_vec)
        
        print("input to encoder size", x.size())
        encoder_seq, memory = self.encoder(x)
        
        encoder_seq = self.encoder_proj(encoder_seq)
        encoder_seq = torch.tanh(encoder_seq)

        print("text encoder output",encoder_seq.size())

        # Change shape
        m_gst = m.view(batch_size, steps, n_mels)
       
        print("mel_size input to Ref encoder", m_gst.size())

        style_embed = self.gst(m_gst, None)
       
        

        # expand style embedding
        #style_embed  = style_embed.unsqueeze(1)

        print("style_embedding output",style_embed.size())
        
     
    
        
        # concatenate
         
        #final_enc_seq = encoder_seq + style_embed
        
        
        

        
         
        
         
        
        #print("final encoded sequence", encoder_seq.size())

        # Need a couple of lists for outputs
        mel_outputs, txt_attn_scores, style_attn_scores = [], [], []
        
               
        #mels, weights = self.decoder(encoder_seq, encoder_seq_proj, prenet_in)
	
               
	# Run the decoder loop
        for t in range(0, steps, self.r) :
           prenet_in = m[:, :, t - 1] if t > 0 else go_frame
           print("prenet input", prenet_in.size()) 
           mel_frames, scores, hidden_states, cell_states, context_vectors = \
               self.decoder(encoder_seq, style_embed, prenet_in, hidden_states, cell_states, context_vectors, t)
           mel_outputs.append(mel_frames)
           txt_scores, style_scores = scores
           txt_attn_scores.append(txt_scores)
           style_attn_scores.append(style_scores)

           
    
       
        #print("XX", mel_outputs[0].size())

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)
        print(mel_outputs.size())
        #print(steps)
       
        #print(mel_outputs[0].size())

  
 

  
        # Post-Process for Linear Spectrograms
        postnet_out, _ = self.postnet(mel_outputs)
       
        
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)
        
        # For easy visualisation
        txt_attn_scores = torch.cat(txt_attn_scores, 1)
        txt_attn_scores = txt_attn_scores.cpu().data.numpy()
            
        style_attn_scores = torch.cat(style_attn_scores, 1)
        style_attn_scores = style_attn_scores.cpu().data.numpy()

        return mel_outputs, linear, txt_attn_scores, style_attn_scores
    
    def generate(self, x,ref_mels,gst_index, steps=2000) :
        print("Generate pass")
        device = next(self.parameters()).device 
        
        self.encoder.eval()
        self.postnet.eval()
        self.gst.eval()
       # self.decoder.eval()
        self.decoder.generating = True
        self.training = False


        if ref_mels is not None:
            print("Ref mel steps defined")
            batch_size, n_mels,m_steps  = ref_mels.size()
        
        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

       
        # Need to initialise all hidden states and pack into tuple for tidyness
        text_attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()
        style_attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()
        attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()

        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        hidden_states = (text_attn_hidden, style_attn_hidden, attn_hidden, rnn1_hidden, rnn2_hidden)
        
        # Need to initialise all lstm cell states and pack into tuple for tidyness
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        cell_states = (rnn1_cell, rnn2_cell)
        
        # Need a <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels).cuda()
        
        # Need an initial context vector
        context_vec_text = torch.zeros(batch_size, self.decoder_dims).cuda()
        context_vec_style = torch.zeros(batch_size, self.decoder_dims).cuda()

        context_vec = torch.zeros(batch_size, self.decoder_dims).cuda()
        context_vectors = (context_vec_text, context_vec_style, context_vec)

        print("Text", x)
        print("Input text size", x.size())
        
        # Project the encoder outputs to avoid 
        # unnecessary matmuls in the decoder loop
        encoder_seq, memory = self.encoder(x)    
        print("encoder sequennce", encoder_seq.size())
   
   

   
        if ref_mels is not None:
            print("Ref mel gst size change")
            m_gst = ref_mels.view(batch_size,m_steps, n_mels)
            print("gst input", m_gst.size())
            
        else:
            m_gst = None

        
        style_embed = self.gst(m_gst, None)
        print("style_embedding", style_embed.size())
         
       # style_embed  = style_embed.unsqueeze(1)

        #print("style embed expanded" , style_embed.size())

        #final_encoder_seq = encoder_seq + style_embed
        
        #encoder_seq_proj = final_encoder_seq
        #encoder_seq_proj = self.encoder_proj_gen(encoder_seq)

        #print("final encoder seq", encoder_seq_proj)
        
        
        # Need a couple of lists for outputs
        mel_outputs,  txt_attn_scores, style_attn_scores  = [], [], []
        #print(batch_size)
         

        # Run the decoder loop
        for t in range(0, steps, self.r) :
            # Passing only the last frame - could pass all frames
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            print(prenet_in)
            mel_frames, scores, hidden_states, cell_states, context_vectors= \
            self.decoder(encoder_seq, style_embed, prenet_in, 
                         hidden_states, cell_states, context_vectors, t)
            mel_outputs.append(mel_frames)
            txt_scores, style_scores = scores
            txt_attn_scores.append(txt_scores)
            style_attn_scores.append(style_scores)
            print(txt_scores)
            print(style_scores)

            context_vec1, context_vec2, f_context = context_vectors
            #print(context_vec1)
            #print(context_vec2)
            print(f_context)
            print(f_context.size())
            
            # Stop the loop if silent frames present
            if (mel_frames < -3.8).all() and t > 10 : break
       
        #print("HERE!") 
        #print("XX", mel_outputs[0])        
       # print("d1:", len(mel_outputs))
       # print("d2", len(mel_outputs[0]))
       # print("d3:", len(mel_outputs[0][0]))
        
        
    
        #print("d4", mel_outputs[0][0][0][0])
        #print("d4", mel_outputs[1][0][0:3])
        #print("d4", mel_outputs[2][0][0:3])

        #print("Concatenate")
        
        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)
                 
        
        
        #print("d5", len(mel_outputs))
        #print("d6", len(mel_outputs[0]))
        #print("d7", len(mel_outputs[0][0]))
        #print("d8", mel_outputs[0][0][0:10])
        

        #print("d3:", mel_outputs.shape)
        
        # Post-Process for Linear Spectrograms
        postnet_out, _ = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
    
    
        linear = linear.transpose(1, 2)[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        
        
        
        # For easy visualisation
       #attn_scores = torch.cat(attn_scores, 1)
       #attn_scores = attn_scores.cpu().data.numpy()[0]
        txt_attn_scores = torch.cat(txt_attn_scores, 1)
        txt_attn_scores = txt_attn_scores.cpu().data.numpy()
            
        style_attn_scores = torch.cat(style_attn_scores, 1)
        style_attn_scores = style_attn_scores.cpu().data.numpy()

        self.encoder.train()
        self.postnet.train()
        self.gst.train()
        self.decoder.generating = False
        

        return mel_outputs, linear, txt_attn_scores, style_attn_scores
    
    
    def init_model(self) :
        for p in self.parameters():
            if p.dim() > 1 : nn.init.xavier_uniform_(p)

    def get_step(self) :
        return self.step.data.item()

    def reset_step(self) :
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    def checkpoint(self, path):
        k_steps = self.get_step() // 1000
        self.save(f'{path}/checkpoint_{k_steps}k_steps.pyt')

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def restore(self, path):
        if not os.path.exists(path):
            print('\nNew Tacotron Training Session...\n')
            self.save(path)
        else:
            print(f'\nLoading Weights: "{path}"\n')
            self.load(path)
            self.decoder.r = self.r.item()

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)

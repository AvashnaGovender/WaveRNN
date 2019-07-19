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

    def __init__(self,ref_enc_filters ):

        super().__init__()
	self.ref_enc_filters
        self.ref_encoder = ReferenceEncoder(self.ref_enc_filters)


        
    def forward(self, inputs):
        
        if inputs is not None:
            print("USING REF")
            style = self.ref_encoder(inputs)      

        return style_embed



class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty, n_mels]  ref mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, ref_enc_filters):

        super().__init__()
        
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        
        self.convs = nn.ModuleList()
        for i in range(K):
            conv = BatchNormConv2D(in_channels=filters[i],out_channels=filters[i + 1] )
            self.convs.append(conv)
        
        
        self.gru = nn.GRU(256,128,batch_first=True)



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

        
        out, mem = self.gru(out)  # mem --- [1, N, E//2]
     
        
       # out =  out[:,-1,:]
        

        return mem.squeeze(0)  #original GST code uses memory and not output

    

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

class BatchNormConv2D(nn.Module) :
    def __init__(self, in_channels, out_channels,  relu=True) :
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = relu

    def forward(self, x) :
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)

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
  
        return x


class PreNet(nn.Module) :
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5) :
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout
        
    def forward(self, x) :
        x = self.fc1(x)
        x = F.relu(x)
        
        x = F.dropout(x, self.p, training=self.training)
        
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
        query_proj = self.W(query).unsqueeze(1)
        
        
        # Compute the scores 
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)
        return scores.transpose(1, 2)



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
        self.attn_net = Attention(decoder_dims) # If attention works procced with attention otherwise need to work with the LSA attention

        self.attn_rnn = nn.GRUCell(decoder_dims+ decoder_dims // 2,decoder_dims)

        
        self.rnn_input = nn.Linear(2 * decoder_dims, lstm_dims)        
        
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)
        
    def zoneout(self, prev, current, p=0.1) :
        mask = torch.zeros(prev.size()).bernoulli_(p).cuda()
        return prev * mask + current * (1 - mask)
    
    def forward(self, encoder_seq, encoder_seq_proj, prenet_in, hidden_states, cell_states, context_vec, t):
        
        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)
        
        
        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states
        
        # PreNet for the Attention RNN
       
        #print("decoder prenet", prenet_in.size())
        prenet_out = self.decoder_prenet(prenet_in)
        
        print("Decoder prenet output size", prenet_out.size())
        
        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)  
        
        print("Attention input full", attn_rnn_in.size())
        
        print("Attention input", attn_rnn_in.squeeze(1).size())
        print("Attention prev", attn_hidden.size())
        
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)
        print("Attention hidden", attn_hidden.size())


        # Compute the attention scores 
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)
        
        print("Decoder attention scores", scores.size())
        
       
        # Dot product to create the context vector 
        context_vec = scores @ encoder_seq
        print("Context Vector", context_vec.size())
        context_vec = context_vec.squeeze(1)
        
        print("Context Vector", context_vec.size())
        
        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1)
        
        
        print("concat context and attention", x.size())

        
        x = self.rnn_input(x)
        

        print("rnn1", x.size())
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
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)
 	
        
        
        print("mel project", mels.size())
        
        
        return mels, scores, hidden_states, cell_states, context_vec
    
    
class Tacotron(nn.Module) :
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, ref_filters) :
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
        self.gst = GST(ref_filters)

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
        attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        
        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        cell_states = (rnn1_cell, rnn2_cell)
        
        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels).cuda()
        
        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims).cuda()      
        
        
        print("input to encoder size", x.size())
        encoder_seq = self.encoder(x)
        
        
        print("text encoder output",encoder_seq.size())

        # Change shape
        m_gst = m.view(batch_size, steps, n_mels)
       
        print("mel_size input to Ref encoder", m_gst.size())

        style_embed = self.gst(m_gst)
       
        print("style_embedding output", style_embed.size())

        # expand style embedding
        style_embed  = style_embed.unsqueeze(1)

        print(style_embed.size())
        
	style_embed = style_embed.expand_as(encoder_seq)        
	
	# add embeddings

        final_enc_seq = encoder_seq + style_embed
        
        
        print("seq concatenate",final_enc_seq.size())

        #encoder_seq_proj = self.encoder_proj_gen(encoder_seq)
        encoder_seq_proj = final_enc_seq 
        print("final encoder input  embedding", encoder_seq_proj.size())
         
        
        #print("final encoded sequence", encoder_seq.size())

        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []
        
               
        #mels, weights = self.decoder(encoder_seq, encoder_seq_proj, prenet_in)
	
               
	# Run the decoder loop
        for t in range(0, steps, self.r) :
           prenet_in = m[:, :, t - 1] if t > 0 else go_frame
           print("prenet input", prenet_in.size()) 
           mel_frames, scores, hidden_states, cell_states, context_vec = \
               self.decoder(final_enc_seq, encoder_seq_proj, prenet_in,hidden_states, cell_states, context_vec, t)
           mel_outputs.append(mel_frames)
           attn_scores.append(scores)
           
    
       

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)
        print(mel_outputs.size())

  
        # Post-Process for Linear Spectrograms
        postnet_out, _ = self.postnet(mel_outputs)
       
        
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)
        
        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()
            
        return mel_outputs, linear, attn_scores
    
    def generate(self, x,ref_mels, steps=2000) :
        print("Generate pass")
        	
        self.encoder.eval()
        self.postnet.eval()
        self.gst.eval()
        
        self.decoder.generating = True
        self.training = False


        if ref_mels is not None:
            print("Ref mel steps defined")
            batch_size, n_mels,m_steps  = ref_mels.size()
        
        
        
        batch_size = 1
        x = torch.LongTensor(x).unsqueeze(0).cuda()
       
        # Need to initialise all hidden states and pack into tuple for tidyness
        attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        
        # Need to initialise all lstm cell states and pack into tuple for tidyness
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        cell_states = (rnn1_cell, rnn2_cell)
        
        # Need a <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels).cuda()
        
        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims).cuda()
        
        print("Text", x)
        print("Input text size", x.size())
        
        # Project the encoder outputs to avoid 
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)    
        print("encoder sequennce", encoder_seq.size())
   
   

   
        if ref_mels is not None:
            print("Ref mel gst size change")
            m_gst = ref_mels.view(batch_size,m_steps, n_mels)
            print("gst input", m_gst.size())
            
        else:
            m_gst = None

        
        style_embed = self.gst(m_gst)

        print("style_embedding", style_embed.size())
    
        style_embed  = style_embed.unsqueeze(1)

        print("style embed expanded" , style_embed.size()) # change code to size (do we need to expand to match encoder_seq)

        final_encoder_seq = encoder_seq + style_embed
        
        encoder_seq_proj = final_encoder_seq
        #encoder_seq_proj = self.encoder_proj_gen(encoder_seq)

        print("final encoder seq", encoder_seq_proj)
        
        
        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []
        #print(batch_size)
         
        
        

        # Run the decoder loop
        for t in range(0, steps, self.r) :
            # Passing only the last frame - could pass all frames
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame 
            
            mel_frames, scores, hidden_states, cell_states, context_vec = \
            self.decoder(final_encoder_seq, encoder_seq_proj, prenet_in, 
                         hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            # Stop the loop if silent frames present
            if (mel_frames < -3.8).all() and t > 10 : break
       
        
        mel_outputs = torch.cat(mel_outputs, dim=2)
        
        # Post-Process for Linear Spectrograms
        postnet_out, _ = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
    
    
        linear = linear.transpose(1, 2)[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        
        
        
        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()[0]
        
        self.encoder.train()
        self.postnet.train()
        self.decoder.generating = False
        
        return mel_outputs, linear, attn_scores
    
    
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

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = (B, T, Q)
        # Keys = (B, T, K)
        # Values = (B, T, V)
        # Outputs = lin_comb:(B, T, V)

        # Here we assume Q == K (dot product attention)
        keys = keys.transpose(1, 2)  # (B, T, K) -> (B, K, T)
        energy = torch.bmm(query, keys)  # (B, T, Q) x (B, K, T) -> (B, T, T)
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize
        linear_combination = torch.bmm(energy, values)  # (B, T, T) x (B, T, V) -> (B, T, V)
        return linear_combination


class GRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.conv_rzo = nn.Conv2d(self.input_channels + self.hidden_channels, 3 * self.hidden_channels, self.kernel_size,
                                 1, self.padding)
        self.conv_rz = nn.Conv2d(self.input_channels +self.hidden_channels, 2 * self.hidden_channels, self.kernel_size,
                                 1,self.padding)
        self.conv_h = nn.Conv2d(self.input_channels + self.hidden_channels, self.hidden_channels, self.kernel_size, 1,
                                self.padding)
        self.W_1x1 = nn.Conv2d(2 * self.hidden_channels, self.hidden_channels, 1)
        self.time_decay_h = nn.Parameter(torch.zeros(self.hidden_channels, 1, 1))
        self.time_decay_m = nn.Parameter(torch.zeros(self.hidden_channels, 1, 1))
        self.time_gate = nn.Parameter(torch.zeros(self.hidden_channels, 1, 1))


    def forward(self, x, h ,c ,m):
        combined1 = torch.cat((x, h), dim=1)
        combined2 = torch.cat((x, m), dim=1)
        A = self.conv_rzo(combined1)
        B = self.conv_rz(combined2)

        decay_h = torch.exp(-self.time_decay_h)  # (C,1,1)
        decay_m = torch.exp(-self.time_decay_m)
        gate = torch.sigmoid(self.time_gate)  # (C,1,1)

        (Ar,Az,Ao) =torch.split(A, A.size()[1]//3, dim=1)
        (Br , Bz)= torch.split(B, B.size()[1] // 2, dim=1)
        r = torch.sigmoid(Ar*gate+h*decay_h)  # reset gate
        z = torch.sigmoid(Az*gate+h*decay_h)  # update gate
        o = torch.sigmoid(Ao)
        xr = torch.sigmoid(Br*gate+m*decay_m)
        xz = torch.sigmoid(Bz*gate+m*decay_m)

        gc = torch.tanh(self.conv_h(torch.cat((x,r*c),dim=1)))
        gm = torch.tanh(self.conv_h(torch.cat((x,xr*m),dim=1)))
        c =z * c +(1-z) *gc
        m = xz * m +(1-xz) *gm
        transformed = self.W_1x1(torch.cat((c,m),dim=1))
        h= o * torch.tanh(transformed)
        return h, r, z, xr, xz, c, m, gc, gm,o

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                    Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                    Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda())
        except:
            return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])),
                    Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])),
                    Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])))


class TransGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias, attention_size):
        super(TransGRU, self).__init__()
        self.attention = Attention(attention_size)
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.bias = bias
        self.all_layers = []


        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = GRUCell(self.input_channels[layer], self.hidden_channels[layer], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        all_outputs = []
        
        
        for step in range(steps):
            input = torch.squeeze(x[:, step, :, :, :], dim=1)
            step_outputs = []
            
            for layer in range(self.num_layers):
                if step == 0:
                    (h, c, _) = GRUCell.init_hidden(bsize, self.hidden_channels[layer], (height, width))
                    if layer == 0:
                        (_, _, m) = GRUCell.init_hidden(bsize, self.hidden_channels[layer], (height, width))
                        internal_state.append((h, c, m))
                    else:
                        (_, _, m) = internal_state[layer - 1]
                        internal_state.append((h, c, m))
                else:
                    if layer == 0:
                        (_, _, m) = internal_state[self.num_layers - 1]
                    else:
                        (_, _, m) = internal_state[layer - 1]
                        
                name = 'cell{}'.format(layer)
                (h,c,_) = internal_state[layer]
                input, r, z, xr, xz, new_c, new_m, gc, gm, o = getattr(self, name)(input, h, c, m)
                internal_state[layer] = (input, new_c, new_m)
                step_outputs.append(input)
            
            all_outputs.append(step_outputs[-1])
        
        all_outputs = torch.stack(all_outputs, dim=1)  # [B, T, C, H, W] [B,T,64,32,32]
        

        attended_outputs = []
        for t in range(steps):
            current_output = all_outputs[:, t]  # [B, C, 32, 32]
            query, keys, values = self.get_QKV(current_output, r, z, xr, xz, new_c, new_m, gc, gm, o)
            query = query.view(bsize, -1, height * width)
            keys = keys.view(bsize, -1, height * width)
            values = values.view(bsize, -1, height * width)
            attended = self.attention(query, keys, values)
            attended = attended.view(bsize, -1, height, width)
            attended_outputs.append(attended)
        

        attended_outputs = torch.stack(attended_outputs, dim=1)  # [B, num_patches, 64, 32, 32]
        
        return attended_outputs

    def get_QKV(self, h_states, r_states, z_states, xr_states,xz_states,c_states,m_states,gc_states,gm_states,o_states):
        values = h_states
        query = (r_states + xr_states + c_states +gc_states)/4
        keys =  (r_states + xr_states + m_states +gm_states)/4
        
        return query, keys, values


class TransGRULayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias, num_classes, attenion_size, patch_size=32):
        super(TransGRULayer, self).__init__()
        self.patch_size = patch_size
        self.forward_net = TransGRU(input_channels, hidden_channels, kernel_size, bias, attention_size=attenion_size)
        self.reverse_net = TransGRU(input_channels, hidden_channels, kernel_size, bias, attention_size=attenion_size)
        self.conv = nn.Conv2d(2*hidden_channels[-1], input_channels, kernel_size=1)
        
    def patchify_image(self, x):
        b, c, h, w = x.size()
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(b, -1, c, self.patch_size, self.patch_size) #[b, n_patches, c, ph, pw]
        return patches
    
    def patchify_image_zigzag(self, x):
        b, c, h, w = x.size()
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [b, h_patches, w_patches, c, ph, pw]
        
        h_patches = patches.size(1)
        w_patches = patches.size(2)
        
        zigzag_patches = []
        
        for i in range(h_patches):
            if i % 2 == 0:
                for j in range(w_patches):
                    zigzag_patches.append(patches[:, i, j])
            else:
                for j in range(w_patches-1, -1, -1):
                    zigzag_patches.append(patches[:, i, j])
        
        zigzag_patches = torch.stack(zigzag_patches, dim=1)  # [b, n_patches, c, ph, pw]
        return zigzag_patches
    
    def unpatchify_image(self, patches, original_shape):
        b, n_patches, c, ph, pw = patches.size()
        h, w = original_shape
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        

        patches = patches.view(b, h_patches, w_patches, c, ph, pw)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        unpatchified = patches.view(b, c, h, w)
        return unpatchified
    
    def unpatchify_image_zigzag(self, zigzag_patches, original_shape):
        b, n_patches, c, ph, pw = zigzag_patches.size()
        h, w = original_shape
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        
        patches = torch.zeros(b, h_patches, w_patches, c, ph, pw, device=zigzag_patches.device)
        
        patch_idx = 0
        for i in range(h_patches):
            if i % 2 == 0:
                for j in range(w_patches):
                    patches[:, i, j] = zigzag_patches[:, patch_idx]
                    patch_idx += 1
            else:
                for j in range(w_patches-1, -1, -1):
                    patches[:, i, j] = zigzag_patches[:, patch_idx]
                    patch_idx += 1
        
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous() #[b, c, h_patches, ph, w_patches, pw]
        unpatchified = patches.view(b, c, h, w)
        return unpatchified
    
    def forward(self, x):
        original_shape = (x.size(2), x.size(3))
        
        patches = self.patchify_image_zigzag(x)
        reversed_patches = torch.flip(patches, dims=[1])


        y_forward = self.forward_net(patches)  # [B, T, C, ph, pw]
        y_reverse = self.reverse_net(reversed_patches)  # [B, T, C, ph, pw]
        
        y_forward = self.unpatchify_image_zigzag(y_forward, original_shape)
        y_reverse = self.unpatchify_image_zigzag(y_reverse, original_shape)

        y = torch.cat((y_forward, y_reverse), dim=1)
        y = self.conv(y)
        return y


class SAGRU(nn.Module):
    def __init__(self, input_channels=64, hidden_channels=[64], kernel_size=5, bias=True, num_classes=4,
                 attenion_size=32*32, encoder=None, patch_size=32):
        super(SAGRU, self).__init__()
        self.encoder = encoder
        self.grulayer = TransGRULayer(input_channels, hidden_channels, kernel_size, bias, num_classes, attenion_size, patch_size)

    def forward(self, x):
        x = self.encoder(x)
        y = self.grulayer(x)
        return y
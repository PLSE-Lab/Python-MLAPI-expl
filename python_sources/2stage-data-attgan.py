#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.functional import softmax
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import pandas as pd
from numpy import linalg as LA


# In[ ]:


path = '../input/2stage/sol_set_tat2.txt'
u_H = pd.read_csv(path, sep=" ", header=None).values

# recued model dim; nb of basis in cem setting
nb_rdm = np.shape(u_H)[1]
print("nb_rdm", nb_rdm)
print("u_H", u_H)

path = '../input/2stage/sol_set_test.txt'
u_H_test = pd.read_csv(path, sep=" ", header=None).values

# recued model dim; nb of basis in cem setting
nb_rdm_test = np.shape(u_H_test)[1]
print("nb_rdm_test", nb_rdm_test)
print("u_H_test", u_H_test)


# In[ ]:


plt.plot(u_H[280,:], "b-")
plt.plot(np.mean(u_H, axis = 0), "r-")


# In[ ]:


print("L1 norm mean", np.mean(LA.norm(u_H-np.mean(u_H, axis = 0), axis = 1, ord = 1)))
print("L2 norm mean", np.mean(LA.norm(u_H-np.mean(u_H, axis = 0), axis = 1)))


# In[ ]:


path = '../input/2stage/f_set_tat2.npy'
ffine = np.load(path)
print("ffine", np.shape(ffine))


path = '../input/2stage/f_set_test.npy'
ffine_test = np.load(path)
print("ffine_test", np.shape(ffine_test))

nb_samples = np.shape(ffine)[0]
nb_times = np.shape(ffine)[1]
nb_reduced = 75
# nb_heads will be defined in the multihead section 
nb_takes = 1

a = 100
b = 100
dim_mesh = (a-1)*(b-1)

fcoarse_reshape= np.reshape(ffine, (nb_samples, nb_times, a-1, b-1))
print("fcoarse_reshape", np.shape(fcoarse_reshape))
fcoarse_reshape_torch = torch.tensor(fcoarse_reshape, dtype=torch.float32)


nb_samples_test = np.shape(ffine_test)[0]
nb_times_test = np.shape(ffine_test)[1]
fcoarse_reshape_test= np.reshape(ffine_test, (nb_samples_test, nb_times_test, a-1, b-1))
print("fcoarse_reshape_test", np.shape(fcoarse_reshape_test))


# In[ ]:


np.mean(fcoarse_reshape)


# In[ ]:


# reduce the spatial dim of f (f projected on the fine mesh ) by max pooling
# reshape the pooling output to a vector
# batch_size: nb of samples
# nb_channels: nb of time steps
# dt: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1; can als be nb of samples, nb of time steps, d_reduced
# output shape: nb of samples, nb of time steps, 81 (reduced dim due to the pooling)
# no training included
rand_factor = 1e-06
class PreProcessingData(nn.Module):
    def __init__(self, bo):
        super().__init__()
        self.pool = nn.MaxPool2d(10,stride = 10)
        self.bo = bo
    # q = [q1; q2;...,qn]; dim: # model dim (d_model), # time steps
    # note the values for Q K are all V which is model reduction coeff; 
    # there is no need to generate the random values for Q and K
    # each row is: the query key and vals of a word
    def forward(self, dt, batch_size, d_channels):
        if self.bo:
            dt = self.pool(dt)
            dt = dt.view(batch_size, d_channels, -1)
            return dt+torch.randn(batch_size, d_channels,  dt.size()[-1])*rand_factor
        else:
            return dt+torch.randn(batch_size, d_channels,  dt.size()[-1])*rand_factor

model = PreProcessingData(True)
out = model(fcoarse_reshape_torch, nb_samples, nb_times)  
np.shape(out)


# In[ ]:


# class OneHeadAttention attention
# # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# generate one head of the transformer (one layer)
# output dim: nb_samples, nb of time steps, reduced dim encode for each time step
class OneHeadAttention(nn.Module):
    def __init__(self, v, d_reduced, bo ):
        super().__init__()
        self.v = v
        
        self.preproc = PreProcessingData(bo)
        self.vv = self.preproc(self.v, nb_samples, nb_times)
        
        self.d_model = self.vv.size()[-1]
        
        self.v_linear = nn.Linear(self.d_model, d_reduced)
        self.q_linear = nn.Linear(self.d_model, d_reduced)
        self.k_linear = nn.Linear(self.d_model, d_reduced)
        
    # self.vv: nb samples, nb time steps, 81 (reduced due to the max pooling, no training)
    def forward(self):
          
        v = self.v_linear( self.vv )
        k = self.k_linear( self.vv )
        q = self.q_linear( self.vv )
        qkt = torch.matmul(q, torch.transpose(k, 1, 2))

        sm_qkt = softmax(qkt, 2)

        out = torch.matmul(sm_qkt, v)
        return out

model = OneHeadAttention(fcoarse_reshape_torch, nb_reduced, True )
out = model()  
np.shape(out)


# In[ ]:


# V1 OF HEAD 6 heads attention
# need class OneHeadAttention
# generate multi-head of one layer using OneHeadAttention; the nb of heads is fixed and is equal to 3 in this code
# # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# output size: nb batches, time steps, reduced dim (note, equal to the encode dim of one head)

nb_heads = 6
class MultiHeads(nn.Module):
    def __init__(self, v, d_reduced, d_head, bo):
        super().__init__()
        self.v = v
        self.head1 = OneHeadAttention(self.v, d_reduced, bo)
        self.head2 = OneHeadAttention(self.v, d_reduced, bo)
        self.head3 = OneHeadAttention(self.v, d_reduced, bo)
        self.head4 = OneHeadAttention(self.v, d_reduced, bo)
        self.head5 = OneHeadAttention(self.v, d_reduced, bo)
        self.head6 = OneHeadAttention(self.v, d_reduced, bo)
        
        self.linear = nn.Linear(d_reduced*d_head, d_reduced)
    def forward(self):
        out1 = self.head1()
        out2 = self.head2()
        out3 = self.head3()
        out4 = self.head4()
        out5 = self.head5()
        out6 = self.head6()
        concat_out = torch.cat((out1, out2, out3, out4, out5, out6), dim = -1)
        
        out = self.linear(  concat_out )
        
        return out
model = MultiHeads(fcoarse_reshape_torch, nb_reduced, nb_heads, True )
out = model()  
print(out.size())


# In[ ]:


# # V2 OF HEAD; 1 head attention
# # need class OneHeadAttention
# # generate multi-head of one layer using OneHeadAttention; the nb of heads is fixed and is equal to 3 in this code
# # # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# # output size: nb batches, time steps, reduced dim (note, equal to the encode dim of one head)

# nb_heads = 1
# class MultiHeads(nn.Module):
#     def __init__(self, v, d_reduced, d_head, bo):
#         super().__init__()
#         self.v = v
#         self.head1 = OneHeadAttention(self.v, d_reduced, bo)
        
#         self.linear = nn.Linear(d_reduced*d_head, d_reduced)
#     def forward(self):
#         out1 = self.head1()
# #         out2 = self.head2()
# #         out3 = self.head3()
# #         out4 = self.head4()
# #         out5 = self.head5()
# #         out6 = self.head6()
# #         concat_out = torch.cat((out1, out2, out3, out4, out5, out6), dim = -1)
        
#         out = self.linear(  out1 )
        
#         return out
# model = MultiHeads(fcoarse_reshape_torch, nb_reduced, nb_heads, True )
# out = model()  
# print(out.size())


# In[ ]:


# # V3 OF HEAD; 3 headS attention
# # need class OneHeadAttention
# # generate multi-head of one layer using OneHeadAttention; the nb of heads is fixed and is equal to 3 in this code
# # # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# # output size: nb batches, time steps, reduced dim (note, equal to the encode dim of one head)

# nb_heads = 3
# class MultiHeads(nn.Module):
#     def __init__(self, v, d_reduced, d_head, bo):
#         super().__init__()
#         self.v = v
#         self.head1 = OneHeadAttention(self.v, d_reduced, bo)
#         self.head2 = OneHeadAttention(self.v, d_reduced, bo)
#         self.head3 = OneHeadAttention(self.v, d_reduced, bo)
#         self.linear = nn.Linear(d_reduced*d_head, d_reduced)
        
#     def forward(self):
#         out1 = self.head1()
#         out2 = self.head2()
#         out3 = self.head3()
# #         out4 = self.head4()
# #         out5 = self.head5()
# #         out6 = self.head6()
#         concat_out = torch.cat((out1, out2, out3), dim = -1)

#         out = self.linear(  concat_out )
        
#         return out
# model = MultiHeads(fcoarse_reshape_torch, nb_reduced, nb_heads, True )
# out = model()  
# print(out.size())


# In[ ]:


# # V4 OF HEAD; 3 headS attention; dropout
# # need class OneHeadAttention
# # generate multi-head of one layer using OneHeadAttention; the nb of heads is fixed and is equal to 3 in this code
# # # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# # output size: nb batches, time steps, reduced dim (note, equal to the encode dim of one head)

# nb_heads = 3
# class MultiHeads(nn.Module):
#     def __init__(self, v, d_reduced, d_head, bo):
#         super().__init__()
#         self.v = v
#         self.head1 = OneHeadAttention(self.v, d_reduced, bo)
#         self.head2 = OneHeadAttention(self.v, d_reduced, bo)
#         self.head3 = OneHeadAttention(self.v, d_reduced, bo)
#         self.linear = nn.Linear(d_reduced*d_head, d_reduced)
#         self.dropout = nn.Dropout(p = 0.5)
#     def forward(self):
#         out1 = self.head1()
#         out2 = self.head2()
#         out3 = self.head3()
# #         out4 = self.head4()
# #         out5 = self.head5()
# #         out6 = self.head6()
#         concat_out = torch.cat((out1, out2, out3), dim = -1)

#         out = self.dropout(self.linear(  concat_out ))
        
#         return out
# model = MultiHeads(fcoarse_reshape_torch, nb_reduced, nb_heads, True )
# out = model()  
# print(out.size())


# In[ ]:


# # version 1 of LAYERS OF TRANSFORMERS: 3 layers transformer
# # need class MultiHeads
# # generate many layers; the nb of layers is fixed and is equal to 3
# # # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# class EncoderLayer(nn.Module):
#     def __init__(self, v, d_reduced, d_head, d_take):
#         super().__init__()
#         self.d_head = d_head
#         # 3 layers of multiheads 
#         self.d_reduced = d_reduced
#         self.layer1 = MultiHeads(v, self.d_reduced, self.d_head, True)
        
#         self.d_take = d_take
    
#     # output dim: (nb_batches, nb_time_steps, encode_dims (d_model) )
#     def forward(self):
#         out1 = self.layer1()
        
#         self.layer2 = MultiHeads(out1, self.d_reduced, self.d_head, False)
#         out2 = self.layer2()

#         self.layer3 = MultiHeads(out2, self.d_reduced, self.d_head, False)
#         out3 = self.layer3()
#         # out3 size: nb_samples, nb_time_steps, dim
#         out = out3[:,-self.d_take:,:].view(nb_samples, -1)
#         return out
        
# model = EncoderLayer(fcoarse_reshape_torch, nb_reduced, nb_heads, nb_takes)
# out = model()  


# In[ ]:


# version 2 of LAYERS OF TRANSFORMERS: 1 layers transformer
# need class MultiHeads
# generate many layers; the nb of layers is fixed and is equal to 3
# # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
class EncoderLayer(nn.Module):
    def __init__(self, v, d_reduced, d_head, d_take):
        super().__init__()
        self.d_head = d_head
        # 3 layers of multiheads 
        self.d_reduced = d_reduced
        self.layer1 = MultiHeads(v, self.d_reduced, self.d_head, True)
        
        self.d_take = d_take
    
    # output dim: (nb_batches, nb_time_steps, encode_dims (d_model) )
    def forward(self):
        out1 = self.layer1()
        
        
        # out3 size: nb_samples, nb_time_steps, dim
        out = out1[:,-self.d_take:,:].view(nb_samples, -1)
        return out
        
model = EncoderLayer(fcoarse_reshape_torch, nb_reduced, nb_heads, nb_takes)
out = model()  
print("out", np.shape(out))


# In[ ]:


# # version 1 OF GENERATOR; nb_reduced = 20 and then l1,l2, l3, l4 4 layers of generations to generate from 20, 40, 60, 75
# # global variables;
# dim_G1L1 = 20
# dim_G1L2 = 40
# dim_G1L3 = 60

# dim_D1L4 = 10
# dim_D1L5 = 5



# # output dim: nb samples, dim of reduced encoded
# # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# class Generator(nn.Module):
#     def __init__(self, d_reduced, d_head, d_take):
#         super(Generator, self).__init__()
#         self.d_reduced = d_reduced
#         self.d_head = d_head
#         self.d_take = d_take
        
#         self.l1 = nn.Linear(self.d_take*self.d_reduced,  dim_G1L1)
#         self.l2 = nn.Linear(dim_G1L1,  dim_G1L2)
#         self.l3 = nn.Linear(dim_G1L2,  dim_G1L3)
#         self.l4 = nn.Linear(dim_G1L3,  nb_rdm)
        
        
        
#     def forward(self, v):
#         self.encode = EncoderLayer(v, self.d_reduced, self.d_head, self.d_take)
#         encode_out = self.encode()
#         encode_out = self.l1(encode_out)
#         encode_out = self.l2(encode_out)
#         encode_out = self.l3(encode_out)
#         G1out = self.l4(encode_out)
                                
#         return G1out

# # output dim: nb samples,1
# # input size: nb samples, dim of reduced encoded
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.l10 = nn.Linear(nb_rdm,  nb_rdm)
#         self.l1 = nn.Linear(nb_rdm,  dim_G1L3)
#         self.l2 = nn.Linear(dim_G1L3,  dim_G1L2)
#         self.l3 = nn.Linear(dim_G1L2,  dim_G1L1)
#         self.l31 = nn.Linear(dim_G1L1,  dim_D1L4)
#         self.l4 = nn.Linear(dim_D1L4,  1)
        
#     # input size: nb_samples, nb_basis 
#     def forward(self, x):
#         x = self.l10(x)
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         x = self.l31(x)
#         x = self.l4(x)
#         x = torch.sigmoid(x)
        
#         return x.view(-1, 1)


# # model = Generator(fcoarse_reshape_torch, nb_reduced, nb_heads, nb_takes)
# # Gout = model()  
# # print(Gout.size())
# # modelD = Discriminator()
# # Dout = modelD(Gout)  
# # print(Dout.size())


# In[ ]:


# # version 2 OF GENERATOR; nb_reduced = 40 and then l1,1 layer of generation to generate from 40 to 75
# # global variables;
# dim_G1L1 = 20
# dim_G1L2 = 40
# dim_G1L3 = 60

# dim_D1L4 = 10
# dim_D1L5 = 5



# # output dim: nb samples, dim of reduced encoded
# # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# class Generator(nn.Module):
#     def __init__(self, d_reduced, d_head, d_take):
#         super(Generator, self).__init__()
#         self.d_reduced = d_reduced
#         self.d_head = d_head
#         self.d_take = d_take
        
#         self.l1 = nn.Linear(self.d_take*self.d_reduced,  nb_rdm)
        
        
        
#     def forward(self, v):
#         self.encode = EncoderLayer(v, self.d_reduced, self.d_head, self.d_take)
#         encode_out = self.encode()
#         G1out = self.l1(encode_out)
                                
#         return G1out

# # output dim: nb samples,1
# # input size: nb samples, dim of reduced encoded
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.l10 = nn.Linear(nb_rdm,  nb_rdm)
#         self.l1 = nn.Linear(nb_rdm,  dim_G1L3)
#         self.l2 = nn.Linear(dim_G1L3,  dim_G1L2)
#         self.l3 = nn.Linear(dim_G1L2,  dim_G1L1)
#         self.l31 = nn.Linear(dim_G1L1,  dim_D1L4)
#         self.l4 = nn.Linear(dim_D1L4,  1)
        
#     # input size: nb_samples, nb_basis 
#     def forward(self, x):
#         x = self.l10(x)
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         x = self.l31(x)
#         x = self.l4(x)
#         x = torch.sigmoid(x)
        
#         return x.view(-1, 1)


# # model = Generator(fcoarse_reshape_torch, nb_reduced, nb_heads, nb_takes)
# # Gout = model()  
# # print(Gout.size())
# # modelD = Discriminator()
# # Dout = modelD(Gout)  
# # print(Dout.size())


# In[ ]:


# version 3 OF GENERATOR; ONE HEAD ONLY; nb_reduced = 75 and then 1 layers of generations to generate from 75 to 75; 
# global variables;
dim_G1L1 = 20
dim_G1L2 = 40
dim_G1L3 = 60

dim_D1L4 = 10
dim_D1L5 = 5



# output dim: nb samples, dim of reduced encoded
# input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
class Generator(nn.Module):
    def __init__(self, d_reduced, d_head, d_take):
        super(Generator, self).__init__()
        self.d_reduced = d_reduced
        self.d_head = d_head
        self.d_take = d_take
        
#         self.l1 = nn.Linear(self.d_take*self.d_reduced,  dim_G1L1)
#         self.l2 = nn.Linear(dim_G1L1,  dim_G1L2)
#         self.l3 = nn.Linear(dim_G1L2,  dim_G1L3)
        self.l4 = nn.Linear(self.d_take*self.d_reduced,  nb_rdm)
        
        
        
    def forward(self, v):
        self.encode = EncoderLayer(v, self.d_reduced, self.d_head, self.d_take)
        encode_out = self.encode()
        G1out = self.l4(encode_out)
                                
        return G1out

# output dim: nb samples,1
# input size: nb samples, dim of reduced encoded
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.l10 = nn.Linear(nb_rdm,  nb_rdm)
        self.l1 = nn.Linear(nb_rdm,  dim_G1L3)
        self.l2 = nn.Linear(dim_G1L3,  dim_G1L2)
        self.l3 = nn.Linear(dim_G1L2,  dim_G1L1)
        self.l31 = nn.Linear(dim_G1L1,  dim_D1L4)
        self.l4 = nn.Linear(dim_D1L4,  1)
        
    # input size: nb_samples, nb_basis 
    def forward(self, x):
        x = self.l10(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l31(x)
        x = self.l4(x)
        x = torch.sigmoid(x)
        
        return x.view(-1, 1)


# model = Generator(fcoarse_reshape_torch, nb_reduced, nb_heads, nb_takes)
# Gout = model()  
# print(Gout.size())
# modelD = Discriminator()
# Dout = modelD(Gout)  
# print(Dout.size())


# In[ ]:


# version 4 OF GENERATOR; ONE or THREE HEADs; nb_reduced = 75 and then 2 layers of generations to generate from 75 to 75, I.E., 75-->75-->75; CORESP TO V2 OF HEAD
# global variables;
dim_G1L1 = 20
dim_G1L2 = 40
dim_G1L3 = 60

dim_D1L4 = 10
dim_D1L5 = 5



# output dim: nb samples, dim of reduced encoded
# input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
class Generator(nn.Module):
    def __init__(self, d_reduced, d_head, d_take):
        super(Generator, self).__init__()
        self.d_reduced = d_reduced
        self.d_head = d_head
        self.d_take = d_take
        
#         self.l1 = nn.Linear(self.d_take*self.d_reduced,  dim_G1L1)
#         self.l2 = nn.Linear(dim_G1L1,  dim_G1L2)
#         self.l3 = nn.Linear(dim_G1L2,  dim_G1L3)
        self.l4 = nn.Linear(self.d_take*self.d_reduced,  nb_rdm)
        self.l5 = nn.Linear(nb_rdm,  nb_rdm)
        
        
        
    def forward(self, v):
        self.encode = EncoderLayer(v, self.d_reduced, self.d_head, self.d_take)
        encode_out = self.encode()
        encode_out = self.l4(encode_out)
        G1out = self.l5(encode_out)
                                
        return G1out

# output dim: nb samples,1
# input size: nb samples, dim of reduced encoded
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.l10 = nn.Linear(nb_rdm,  nb_rdm)
        self.l1 = nn.Linear(nb_rdm,  dim_G1L3)
        self.l2 = nn.Linear(dim_G1L3,  dim_G1L2)
        self.l3 = nn.Linear(dim_G1L2,  dim_G1L1)
        self.l31 = nn.Linear(dim_G1L1,  dim_D1L4)
        self.l4 = nn.Linear(dim_D1L4,  1)
        
    # input size: nb_samples, nb_basis 
    def forward(self, x):
        x = self.l10(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l31(x)
        x = self.l4(x)
        x = torch.sigmoid(x)

        return x.view(-1, 1)



# model = Generator(fcoarse_reshape_torch, nb_reduced, nb_heads, nb_takes)
# Gout = model()  
# print(Gout.size())
# modelD = Discriminator()
# Dout = modelD(Gout)  
# print(Dout.size())


# In[ ]:


# # version 5 OF GENERATOR; ONE or THREE HEADs; nb_reduced = 75 and then 2 layers of generations to generate from 75 to 75, I.E., 75-->75-->75; CORESP TO V2 OF HEAD
# # dropout at generator and combining heads 
# # global variables;
# dim_G1L1 = 20
# dim_G1L2 = 40
# dim_G1L3 = 60

# dim_D1L4 = 10
# dim_D1L5 = 5



# # output dim: nb samples, dim of reduced encoded
# # input dim: (500, 15, 99, 99) nb samples, nb time steps, a-1, b-1
# class Generator(nn.Module):
#     def __init__(self, d_reduced, d_head, d_take):
#         super(Generator, self).__init__()
#         self.d_reduced = d_reduced
#         self.d_head = d_head
#         self.d_take = d_take
        
# #         self.l1 = nn.Linear(self.d_take*self.d_reduced,  dim_G1L1)
# #         self.l2 = nn.Linear(dim_G1L1,  dim_G1L2)
# #         self.l3 = nn.Linear(dim_G1L2,  dim_G1L3)
#         self.l4 = nn.Linear(self.d_take*self.d_reduced,  nb_rdm)
#         self.l5 = nn.Linear(nb_rdm,  nb_rdm)
#         self.dropout = nn.Dropout(p = 0.5)
        
        
#     def forward(self, v):
#         self.encode = EncoderLayer(v, self.d_reduced, self.d_head, self.d_take)
#         encode_out = self.encode()
#         encode_out = self.dropout(self.l4(encode_out))
#         G1out = self.dropout(self.l5(encode_out))
                                
#         return G1out

# # output dim: nb samples,1
# # input size: nb samples, dim of reduced encoded
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.l10 = nn.Linear(nb_rdm,  nb_rdm)
#         self.l1 = nn.Linear(nb_rdm,  dim_G1L3)
#         self.l2 = nn.Linear(dim_G1L3,  dim_G1L2)
#         self.l3 = nn.Linear(dim_G1L2,  dim_G1L1)
#         self.l31 = nn.Linear(dim_G1L1,  dim_D1L4)
#         self.l4 = nn.Linear(dim_D1L4,  1)
        
#     # input size: nb_samples, nb_basis 
#     def forward(self, x):
#         x = self.l10(x)
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         x = self.l31(x)
#         x = self.l4(x)
#         x = torch.sigmoid(x)

#         return x.view(-1, 1)



# # model = Generator(fcoarse_reshape_torch, nb_reduced, nb_heads, nb_takes)
# # Gout = model()  
# # print(Gout.size())
# # modelD = Discriminator()
# # Dout = modelD(Gout)  
# # print(Dout.size())


# In[ ]:


# prepare the training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

u_H_torch = torch.tensor(u_H, dtype=torch.float32, device = device)
fcoarse_reshape_torch = torch.tensor(fcoarse_reshape, dtype=torch.float32, device = device)

lr = 0.0003
beta1 = 0.5

netG = Generator(nb_reduced, nb_heads, nb_takes).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# In[ ]:


# train D to get a better D first
epochs = 100
for ep in range(epochs):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train D with real; input u_H dim: batch size, nb basis
    netD.zero_grad()
    labels = torch.full((nb_samples, 1), 1, device=device)
    
    output = netD(u_H_torch)
    errD_real = criterion(output, labels)
    errD_real.backward()
    # the average output (across the batch) of the discriminator for the all real batch. 
    # This should start close to 1 then theoretically converge to 0.5 when G gets better. 
    D_x = output.mean().item()
    
    # train with fake
    fake = netG(fcoarse_reshape_torch)
    labels.fill_(0)
    # detach will reqiure NO gradient
    
    output = netD(fake.detach())
    errD_fake = criterion(output, labels)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()
    
    if ep % 10 == 0:
        print('[%d] Loss_D: %.4f D(x): %.4f D(G(z)): %.4f'
                  % (ep, errD.item(), D_x, D_G_z1))


# In[ ]:


# training
epochs = 2500
loss_D_set = []
loss_G_set = []
for ep in range(epochs):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train D with real; input u_H dim: batch size, nb basis
    netD.zero_grad()
    labels = torch.full((nb_samples, 1), 1, device=device)
    output = netD(u_H_torch)
    errD_real = criterion(output, labels)
    errD_real.backward()
    # the average output (across the batch) of the discriminator for the all real batch. 
    # This should start close to 1 then theoretically converge to 0.5 when G gets better. 
    D_x = output.mean().item()
    
    # train with fake
    
    fake = netG(fcoarse_reshape_torch)
    labels.fill_(0)
    # detach will reqiure NO gradient
    output = netD(fake.detach())
    errD_fake = criterion(output, labels)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    loss_D_set.append(errD.item())
    optimizerD.step()
    
    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    labels.fill_(1)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D    
    # Calculate G's loss based on this output
    output = netD(fake)
    errG = criterion(output, labels)
    loss_G_set.append(errG.item())
    errG.backward()
    # average discriminator outputs for the all fake batch. 
    # The first number is before D is updated and the second number is after D is updated. 
    # These numbers should start near 0 and converge to 0.5 as G gets better. 
    D_G_z2 = output.mean().item()
    
    optimizerG.step()

    if ep % 100 == 0:
        print('[%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (ep, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


# In[ ]:


plt.plot(loss_D_set)
plt.show()
plt.plot(loss_G_set)
plt.show()


# In[ ]:


outG = np.array(netG(fcoarse_reshape_torch).detach())
plt.plot(u_H[100,:], "r-")
plt.plot(outG[100,:], "b-")


# In[ ]:


print("L1 norm training", np.mean(LA.norm(u_H-outG, axis = 1, ord = 1)))
print("L2 norm training", np.mean(LA.norm(u_H-outG, axis = 1)))


# In[ ]:


u_H_test_torch = torch.tensor(u_H_test, dtype=torch.float32, device = device)
fcoarse_reshape_test_torch = torch.tensor(fcoarse_reshape_test, dtype=torch.float32, device = device)
outG_test = np.array(netG(fcoarse_reshape_test_torch).detach())
print("L1 norm testing", np.mean(LA.norm(u_H_test-outG_test, axis = 1, ord = 1)))
print("L2 norm testing", np.mean(LA.norm(u_H_test-outG_test, axis = 1)))


# 1. $6$ heads, $1$ layer and nb of reduced is equal to $20$. 
# $L_1$ norm testing $5.943024160929824$
# $L_2$ norm testing $0.843793466442161$
# 
# 2. $3$ heads, $1$ layer and nb of reduced is equal to $40$; then $40, 60, 75$.
# L1 norm testing 3.6575742146939154
# L2 norm testing 0.5693827955547961
# 
# 3. $3$ heads, $1$ layer and nb of reduced is equal to $40$; then $40, 50, 60, 70, 75$.
# L1 norm testing 5.892637887206401
# L2 norm testing 0.8733909057241649
# 
# 4. $1$ head, $1$ layer and nb of reduced is equal to $75$; then ONE LAYER $75, 75$.
# L1 norm testing 4.457730482973083
# L2 norm testing 0.7144304588711031
# 
# 4.1 $6$ head (difference with 4), $1$ layer and nb of reduced is equal to $75$; then ONE LAYER $75, 75$.
# Compare to 4; the training is not good; seems that longer training will make this better
# L1 norm training 3.3722966909764525
# L2 norm training 0.5721519362195117
# 
# 5. $1$ head, $1$ layer and nb of reduced is equal to $75$; then TWO LAYER $75-->75-->75$.
# L1 norm testing 3.545499672931557
# L2 norm testing 0.549414816966618
# 
# 6. $3$ head, $1$ layer and nb of reduced is equal to $75$; then TWO LAYER $75-->75-->75$.
# L1 norm testing 3.237379606326087
# L2 norm testing 0.4942014560515833
# 
# 7. $3$ head, $1$ layer and nb of reduced is equal to $75$; then TWO LAYER $75-->75-->75$.
# ReLUs are used at combining heads and frist 75 ---> 75 linear layer
# L1 norm testing 3.376117860343691
# L2 norm testing 0.5087644974323288
# 
# 8. $3$ head, $1$ layer and nb of reduced is equal to $75$; then TWO LAYER $75-->75-->75$.
# dropouts are used at combining heads and frist 75 ---> 75 linear layer
# L1 norm testing 8.110079345427293
# L2 norm testing 1.4909529028322355
# Not sure if failed due to the mis-use of the dropout function
# 
# 
# 7. $3$ head, $1$ layer and nb of reduced is equal to $75$; 
# two layers after concat; from d_reduced*d_head ---> 2*d_reduced ---> d_reduced
# then TWO LAYER $75-->75-->75$
# 
# ReLUs are used at combining heads and frist 75 ---> 75 linear layer
# L1 norm testing 3.1449419005253083
# L2 norm testing 0.5011790415401082
# 
# 8.$3$ head, $1$ layer and nb of reduced is equal to $75$; compared to 7 (pooling are different)
# two layers after concat; from d_reduced*d_head ---> 2*d_reduced ---> d_reduced
# 
# then TWO LAYER $75-->75-->75$
# AVERAGE POOL (difference with 7)
# L1 norm testing 3.1958206570736762
# L2 norm testing 0.5000293122129135
# 
# 4.2 $6$ head (difference with 4), $1$ layer and nb of reduced is equal to $75$; then ONE LAYER $75, 75$.
# Compare to 4; the training is not good; seems that longer training will make this better; 2000 epochs 
# L1 norm testing 2.949004405407101
# L2 norm testing 0.4740920027359995
# 
# 9. 6 heads; concate from 6*75--->3*75--->2*75--->1*75; one layer afterwards (75*75)
# L1 norm testing 3.766360662973743
# L2 norm testing 0.6293451880682172
# 
# 9.1
# 6 heads; concate from 6*75--->3*75--->2*75--->1*75; one layer afterwards (75*75) 2000 epochs (ReLU)
# L1 norm testing 3.5053833140001087
# L2 norm testing 0.5474841715679664
# 
# 
# 10. $6$ head (difference with 4), $1$ layer and nb of reduced is equal to $75$; then two LAYERs of  $75, 75$. 2500 epochs
# converges fast; the diff with a slow converg one is the addition layer of 75, 75; but this one converges fast but result is not that good
# L1 norm testing 3.9214912109647058
# L2 norm testing 0.5774703958037964
# 
# 
# Failed
# 1. $3$ head, $1$ layer and nb of reduced is equal to $75$; then TWO LAYER $75-->75-->75$. 
# Normalization added to the discriminator; the training for both D(pretraing) and G are dead.
# 
# Conclusion:
# 1. nb_reduced should be relative larger
# 2. generation layers are relatively more important
# 3. nb of heads also makes sense
# 
# 
# 
# 
# 

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from numpy import linalg as LA
import pandas as pd
import time
from scipy.linalg import eigh
from scipy.linalg import eig
from tqdm import tqdm

# load data
path = '../input/full-mesh/ele.txt'
ele = pd.read_csv(path, sep=" ", header=None)
ele = ele.values
ele = ele[:,range(2,6)]

path2 = '../input/full-mesh/coord.txt'
coord = pd.read_csv(path2, sep=" ", header=None)
coord = coord.values


path3 = '../input/full-mesh/ms-data.txt'
ms_data = pd.read_csv(path3, sep = r'\s{2,}', header=None, engine='python')
ms_data = ms_data.values

# nb of elements in x dir
a = 100
# nb of elements in y dir
b = 100

# total dofs
dof = np.unique(ele)
nb_dof = np.size(dof)

# nb of element
nb_ele,nb_col = ele.shape


int_dof_domain = []
for i in range(1, a):
    for j in range(1, b):
        int_dof_domain.append(i*(a+1)+j)


# In[ ]:


cembasis_out = []
path5 = '../input/cem100/cembasis_5.npy'
path10 = '../input/cem100/cembasis_10.npy'
path20 = '../input/cem100/cembasis_20.npy'
cembasis_out.append(np.load(path5))
cembasis_out.append(np.load(path10))
cembasis_out.append(np.load(path20))


# In[ ]:


ms_data1 = ms_data[0:-1:6, 0:-1:6]
plt.imshow(ms_data1)
plt.colorbar()
plt.show()
kappa = np.reshape(ms_data1,(10000,))
for i in range(len(kappa)):
    if kappa[i]>10:
        kappa[i] = 1000
        
print(np.shape(kappa))
print(kappa.min())
print(kappa.max())


# In[ ]:


class mesh:
    # a: nb of fine in x
    # b: nb of fine in y
    # sc: coarse mesh size; i.e., in x direction, how many fine elements are there in
    # in coarse ele
    def __init__(self, a, b, sc):
        # self.a: nb of fine elements in x
        self.a = a
        # self.b: nb of fine elements in y
        self.b = b
        # coarse mesh size 
        self.coarse_size = sc
        # nb of coarse mesh in x
        self.nb_coarse = int(self.a/sc)
       
        
    
    # ext: is the over spl parameter; i.e, in one direction, how many more coarse
    # ele should be taken; if ext = 0, no over sampling;
    # return an 2d array, each row in this array is the 
    # oversampling coarse elements of this coarse elements; the return is sorted
    def split_coarse_eles(self, ext):
        # how many coarse elements to extend?
        self.ext = ext
        self.over_spl = []
        self.size_nbhd = []
        ctr = 0
        for i in range(0, self.nb_coarse):
            for j in range(0, self.nb_coarse):
                xpos = 0
                xneg = 0
                ypos = 0
                yneg = 0 
                
                while i+ypos<self.nb_coarse-1:
                    if ypos<self.ext:
                        ypos = ypos+1
                    else:
                        break
                
                while j+xpos<self.nb_coarse-1:
                    if xpos<self.ext:
                        xpos = xpos+1
                    else:
                        break
                
                while j-xneg>0:
                    if xneg<self.ext:
                        xneg = xneg+1
                    else:
                        break
                
                while i-yneg>0:
                    if yneg<self.ext:
                        yneg = yneg+1
                    else:
                        break
                
                # left bottom corner (first) coarse element id of the over sampling region
                start_k = (i-yneg)*self.nb_coarse+(j-xneg)         
                # right bottom corner coarse element id of the over sampling region
                end_k = (i-yneg)*self.nb_coarse+(j+xpos)
                
                # over sampled coarse blks for coarse block (i,j)
                over_spl_id = []
                for p in range(ypos+yneg+1):
                    for q in range(start_k, end_k+1):
                        over_spl_id.append(q+p*self.nb_coarse)
                    
                self.over_spl.append(over_spl_id)
                self.size_nbhd.append([ yneg+ypos+1, xneg+xpos+1])
                ctr = ctr+1
                
        return self.over_spl, self.size_nbhd
    
    # memory friendly version
    # ext: is the over spl parameter; i.e, in one direction, how many more coarse
    # i, j is the ith row jth col coarse element
    # no global varaible is defined
    # return is the array of all fine eles of the over sampled region
    def mem_split_coarse_eles(self, ext,i,j):
        # how many coarse elements to extend?
        self.ext = ext
        
        xpos = 0
        xneg = 0
        ypos = 0
        yneg = 0 

        while i+ypos<self.nb_coarse-1:
            if ypos<self.ext:
                ypos = ypos+1
            else:
                break

        while j+xpos<self.nb_coarse-1:
            if xpos<self.ext:
                xpos = xpos+1
            else:
                break

        while j-xneg>0:
            if xneg<self.ext:
                xneg = xneg+1
            else:
                break

        while i-yneg>0:
            if yneg<self.ext:
                yneg = yneg+1
            else:
                break
                
        # left bottom corner (first) coarse element id of the over sampling region
        start_k = (i-yneg)*self.nb_coarse+(j-xneg)         
        # right bottom corner coarse element id of the over sampling region
        end_k = (i-yneg)*self.nb_coarse+(j+xpos)

        # over sampled coarse blks for coarse block (i,j)
        over_spl_id = []
        for p in range(ypos+yneg+1):
            for q in range(start_k, end_k+1):
                over_spl_id.append(q+p*self.nb_coarse)
                
        return over_spl_id
    
    # return the left bottom fine element of each (all) coarse ele
    # define the global variable left bottom fine element of each (all) coarse ele
    # the order is from left to right from bottom to top
    def left_bottom_fine_ele(self):
        self.left_bottom_fine = []
        
        first_row = [0+i*self.coarse_size for i in range(self.nb_coarse)]
        for i in range(self.nb_coarse):
            for j in first_row:
                self.left_bottom_fine.append(i*self.a*self.coarse_size+j)
        
        return self.left_bottom_fine
        
        
    # return fine element of a coarse block
    # to run this function, first need to run left_bottom_fine_ele
    # to get left bottom fine element of each coarse ele
    # ix is the index of the this coarse ele; 0 is the first one (left bot one)
    def fine_ele_coarse_block(self, ix):
        # fine ele of coarse block ix
        res = []
        first_row = [self.left_bottom_fine[ix]+i for i in range(self.coarse_size)]
        
        for i in range(self.coarse_size):
            for j in first_row:
                res.append(i*a+j)
        
        
        return res
        
    # return dofs of a coarse ele; 
    # need to use self.left_bottom_fine
    def dof_coarse_ele(self, ele, ix):
        start = min(ele[self.left_bottom_fine[ix]])
        res = []
        first_row = [start+i for i in range(self.coarse_size+1)]
        for i in range(self.coarse_size+1):
            for j in first_row:
                res.append(i*(a+1)+j)
        return res
    
    # boundary dofs of a coarse ele
    # ix is the index of the coarse element; ix = 0, is the first one
    # return is already sorted
    def bnd_dof_coarse_ele(self, ele, ix):
        start = min(ele[self.left_bottom_fine[ix]])
        res = []
        for i in range(self.coarse_size+1):
            res.append(start+i)
        
        for i in range(1, self.coarse_size):
            res.append(start+(a+1)*i)
            res.append(start+(a+1)*i+self.coarse_size)
        
        for i in range(0, self.coarse_size+1):
            res.append(start+(a+1)*(self.coarse_size)+i)
        
        return res
    
    # return the int dofs of a coarse ele
    # ix is the index of the coarse ele
    def int_dof_coarse_ele(self, ele, ix):
        start = min(ele[self.left_bottom_fine[ix]])+a+1+1
        
        first_row = [start+i for i in range(self.coarse_size-1)]
        
        res = []
        
        for i in range(self.coarse_size-1):
            for j in first_row:
                res.append(i*(a+1)+j)
        
        return res
        
    # return int dof of a coarse nbhd
    # need to use self.over_spl and self.left_bottom_fine
    # ix is the coarse nbhd id; 0 is the first coarse nbhd(generated from first coarse ele)
    def int_dof_coarse_nbhd(self,ele, ix):
        # the first coarse ele of the coarse nbhd ix
        start_ele = self.over_spl[ix][0]
        
        r1 = self.size_nbhd[ix][0]
        c1 = self.size_nbhd[ix][1]
        res = []
        start_dof = min(ele[self.left_bottom_fine[start_ele]])+a+1+1
        
        first_row = [start_dof+i for i in range(c1*self.coarse_size-1)]
        
        for i in range(self.coarse_size*r1-1):
            for j in first_row:
                res.append(i*(a+1)+j)
        
        return res
    
    def fine_ele_coarse_nbhd(self, ix):
        start_ele = self.over_spl[ix][0]
        r1 = self.size_nbhd[ix][0]
        c1 = self.size_nbhd[ix][1]
        res = []
        
        # starting fine element of the coarse ele start_ele
        start_fine = self.left_bottom_fine[start_ele]
        for i in range(r1*self.coarse_size):
            for j in range(c1*self.coarse_size):
                res.append(start_fine+i*a+j)
        
        return res
                
    
    
    # return the bnd dof of a coarse nbhd
    # need to use self.over_spl and self.left_bottom_fine
    def bnd_dof_coarse_nbhd(self, ele, ix):
        start_ele = self.over_spl[ix][0]
        
        r1 = self.size_nbhd[ix][0]
        c1 = self.size_nbhd[ix][1]
        res = []
        start_dof = min(ele[self.left_bottom_fine[start_ele]])
        
        first_row = [start_dof+i for i in range(c1*self.coarse_size+1)]
        res = res+first_row
        for i in range(1, self.coarse_size*r1):
            res.append(start_dof+i*(a+1))
            res.append(first_row[-1]+i*(a+1))
        
        for j in first_row:
            res.append(j+self.coarse_size*r1*(a+1))
        
        return res
            
    def dof_coarse_nbhd(self, ele, ix):
        # the first coarse ele of the coarse nbhd ix
        start_ele = self.over_spl[ix][0]
        
        r1 = self.size_nbhd[ix][0]
        c1 = self.size_nbhd[ix][1]
        res = []
        start_dof = min(ele[self.left_bottom_fine[start_ele]])
        
        first_row = [start_dof+i for i in range(c1*self.coarse_size+1)]
        
        for i in range(self.coarse_size*r1+1):
            for j in first_row:
                res.append(i*(a+1)+j)
        return res

d = mesh(100,100,5)


# In[ ]:


loc_stiff = [[  2/3, -1/6, -1/3, -1/6],
            [ -1/6,  2/3, -1/6, -1/3],
             [ -1/3, -1/6,  2/3, -1/6],
            [ -1/6, -1/3, -1/6,  2/3]
            ]

loc_stiff = np.matrix(loc_stiff)

p1p1 = 1/90000
p1p2 = 1/180000
p1p3 = 1/360000
p1p4 = 1/180000

p2p2 = 1/90000
p2p3 = 1/180000
p2p4 = 1/360000

p3p3 = 1/90000
p3p4 = 1/180000

p4p4 = 1/90000

loc_mass = [[p1p1,p1p2,p1p3,p1p4],
        [p1p2,p2p2,p2p3,p2p4],
        [p1p3,p2p3,p3p3,p3p4],
        [p1p4,p2p4,p3p4,p4p4]]

loc_mass =np.matrix(loc_mass)


# In[ ]:


# part 1/3
# define global fine stiff and mass matrix
# and hence define _RtMR_ and _RtSR_
# these two matrices are indep of coarse mesh size
# make global mass and stiff matrix
# domain bnd dofs are included
stiff_fine = np.zeros(( len(dof), len(dof) ))
mass_fine = np.zeros((  len(dof), len(dof) ))
mass_norm = np.zeros((  len(dof), len(dof) ))

for jx in range(len(ele)):
    loc_dof = ele[jx]
    coeff = kappa[jx] 
    for p in range(0,4):
        for q in range(0,4):
            ai = loc_dof[p]
            bi = loc_dof[q]
            stiff_fine[ai,bi] = stiff_fine[ai,bi]+coeff*loc_stiff[p,q]
            mass_fine[ai,bi] = mass_fine[ai,bi]+loc_mass[p,q]
            mass_norm[ai,bi] = mass_norm[ai,bi]+coeff*loc_mass[p,q]


stiff_fine = stiff_fine[int_dof_domain ,:]
stiff_fine = stiff_fine[:, int_dof_domain]

mass_fine = mass_fine[int_dof_domain ,:]
mass_fine = mass_fine[:, int_dof_domain]

mass_norm = mass_norm[int_dof_domain ,:]
mass_norm = mass_norm[:, int_dof_domain]


# part 2/3
# the following: for f
# f = sin(pi x)sin(pi y)
x = np.zeros(a+1)
for i in range(0,a+1):
    x[i] = 0+i*1/a
y = np.zeros(b+1)
for i in range(0,b+1):
    y[i] = 0+i*1/b

xv, yv = np.meshgrid(x,y)
f_notime = np.multiply(np.sin(np.pi*xv), np.sin(np.pi*yv))
f_notime = f_notime.reshape((a+1)*(b+1))
f_notime = f_notime[int_dof_domain]

print("f_notime",np.shape(f_notime))

# will be used in coarse solution time evolution
Massf_notime = np.matmul(mass_fine, f_notime)


# In[ ]:


# for the purpose of calculating the fine exact solution
invM = LA.inv(mass_fine)
invMS = np.matmul(invM, stiff_fine)


# In[ ]:


# true solution: sin(t)sin(pi x)sin(pi y)
# to be cfl correct: dt should be less than 0.0001
Tt = 4.0

# 0.00001 has been used in the previous tests, works for alpha = 0.001
dt = 0.00001


nt = int(Tt/dt+0.5)

# alpha_set = [0.01, 0.05, 0.1 0.5, 1.0, 5, 10]

# teleport
# I have tried 0.01 and 10
alpha_set = [0.1]


exact_sol_alpha = []

# each entry is the norm for one alpha
exact_stiff_energy = []
exact_mass_l2 = []


for alpha in alpha_set:
    print("alpha ------------------------------------------------------->: ", alpha)
    print("dt: ", dt)

    coeff_np1 = alpha/(dt*dt)+1/(2*dt)
    coeff_n = 2*alpha/(dt*dt)
    coeff_nm1 = 1/(2*dt)-alpha/(dt*dt)

    coeff_n_np1 = coeff_n/coeff_np1
    coeff_nm1_np1 = coeff_nm1/coeff_np1


    # define initial condition; map from fine scale solution to coarse sclae solution
    fine_u0 = np.zeros((a-1)*(b-1))
    fine_u1 = np.zeros((a-1)*(b-1))


    # find the exact solution using very fine mesh
    print("start to evaluate the fine solution")
    for tx in tqdm(range(1, nt+1)):
        lhs = coeff_n_np1*fine_u1+coeff_nm1_np1*fine_u0-1/coeff_np1*np.matmul(invMS, fine_u1)+1/coeff_np1*f_notime*np.sin(np.pi*tx*dt)
        fine_u0 = fine_u1
        fine_u1 = lhs


    exact_sol_alpha.append(lhs)
    exact_mass_l2.append( np.matmul(     lhs     ,np.matmul(  mass_norm, lhs )    ) )
    exact_stiff_energy.append( np.matmul(     lhs     ,np.matmul(  stiff_fine, lhs )    ) )


# In[ ]:


# need to define kappa, nb_basis, c1, c2, loc_mass, loc_stiff
# define R_off and T_off, two global matrices
# R_off: nb of fine dofs in a coarse element, nb of aux basis in a coarse element
# T_off: nb of coarse elements, nb of aux basis in an element, nb of fine dofs in a coarse element
nb_basis = 3
Ex = [6, 4 ,3]

# each entry is a [], in which is the error for different alphas for given coarse nbhd size
err_decay_stiff = []
err_decay_mass = []
err_decay_stiff_ratio = []
err_decay_mass_ratio = []
delta_t = []
Hconv = [5, 10, 20]

# loop over all extension
for hx, Hsize in enumerate(Hconv):
    
    
    cembasis = cembasis_out[hx]
    
    # mass_fine and stiff_fine are defined in the previous block
    _RtMR_ = np.matmul(np.transpose(cembasis),np.matmul(mass_fine, cembasis))
    _RtSR_ = np.matmul(np.transpose(cembasis),np.matmul(stiff_fine, cembasis))
    
    invRtMR = LA.inv(_RtMR_)
    invMS_cem = np.matmul(invRtMR, _RtSR_)
    fcoarse = np.matmul( invRtMR    , np.matmul( np.transpose(cembasis) , Massf_notime)  )
    
    # fixed coarse element size; each entry is the error for one alpha
    err_decay_stiff_alpha = []
    err_decay_mass_alpha = []
    err_decay_stiff_alpha_ratio = []
    err_decay_mass_alpha_ratio = []
    
    # loop over all alpha given a fixed coarse element size
    # cem basis is uniform for all alpha
    ax = 0
    for alpha in alpha_set: 
        print("alpha ------------------------------------------------------->: ", alpha)
        print("dt: ", dt)

        coeff_np1 = alpha/(dt*dt)+1/(2*dt)
        coeff_n = 2*alpha/(dt*dt)
        coeff_nm1 = 1/(2*dt)-alpha/(dt*dt)

        coeff_n_np1 = coeff_n/coeff_np1
        coeff_nm1_np1 = coeff_nm1/coeff_np1


        # define initial condition; map from fine scale solution to coarse sclae solution
        fine_u0 = np.zeros((a-1)*(b-1))
        fine_u1 = np.zeros((a-1)*(b-1))
        
        
        tla, tlb = np.shape(cembasis)
        
        # coarse solution at time 0 and time dt
        coarse_u0 = np.zeros(tlb)
        coarse_u1 = np.zeros(tlb)
        
        for tx in tqdm(range(1, nt+1)):
            lhs = coeff_n_np1*coarse_u1+coeff_nm1_np1*coarse_u0-1/coeff_np1*np.matmul(invMS_cem, coarse_u1)+1/coeff_np1*fcoarse*np.sin(np.pi*tx*dt)
            coarse_u0 = coarse_u1
            coarse_u1 = lhs
        
        # calculating the loss
        err = np.matmul( cembasis , lhs)- exact_sol_alpha[ax]
        ratio_stiff = np.matmul(np.transpose(err), np.matmul( stiff_fine ,  err  ))
        ratio_mass = np.matmul(np.transpose(err), np.matmul( mass_norm ,  err  ))
        
        err_decay_stiff_alpha.append(ratio_stiff)
        err_decay_mass_alpha.append(ratio_mass)
        
        err_decay_stiff_alpha_ratio.append(ratio_stiff/exact_stiff_energy[ax])
        err_decay_mass_alpha_ratio.append(ratio_mass/exact_mass_l2[ax])
        
        # increment the loop index for alpha
        ax = ax+1
        
    err_decay_stiff.append(err_decay_stiff_alpha)
    err_decay_mass.append(err_decay_mass_alpha)
    
    err_decay_stiff_ratio.append(err_decay_stiff_alpha_ratio)
    err_decay_mass_ratio.append(err_decay_mass_alpha_ratio)


# In[ ]:


print("err_decay_stiff", err_decay_stiff)
print("err_decay_mass", err_decay_mass)
np.savetxt("err_decay_stiff.txt",err_decay_stiff)
np.savetxt("err_decay_mass.txt",err_decay_mass)
err_decay_mass_ay = np.array(err_decay_mass)
err_decay_stiff_ay = np.array(err_decay_stiff)


print("err_decay_stiff_ratio", err_decay_stiff_ratio)
print("err_decay_mass_ratio", err_decay_mass_ratio)
np.savetxt("err_decay_stiff_ratio.txt",err_decay_stiff_ratio)
np.savetxt("err_decay_mass_ratio.txt",err_decay_mass_ratio)
err_decay_mass_ay_ratio = np.array(err_decay_mass_ratio)
err_decay_stiff_ay_ratio = np.array(err_decay_stiff_ratio)


# In[ ]:


for ax in range(len(alpha_set)):
    print("alpha----------------------------------------------------------> : ", alpha_set[ax])
    err_decay_stiff = err_decay_stiff_ay[:, ax]
    print("err_decay_stiff: ", err_decay_stiff, "exact: ",  exact_stiff_energy[ax])
    print("stiff norm (a) error ratio: ",100*err_decay_stiff_ay_ratio[:, ax], "%")
    
    plt.plot( np.log(   err_decay_stiff_ay_ratio[:, ax]   )  )  
    
    plt.title("stiff (energy) norm")
plt.show()
    


# In[ ]:


for ax in range(len(alpha_set)):
    print("alpha----------------------------------------------------------> : ", alpha_set[ax])
    err_decay_mass = err_decay_mass_ay[:, ax]
    print("err_decay_mass: ", err_decay_mass, "exact: ",  exact_mass_l2[ax])
    print("mass norm (a) error relative: ", 100*err_decay_mass_ay_ratio[:, ax], "%")
    plt.plot((np.log(   err_decay_mass_ay_ratio[:, ax]     ) )  )
    plt.title("mass (l2) norm")
plt.show()
    


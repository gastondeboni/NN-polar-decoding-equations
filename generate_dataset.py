#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:24:46 2024
@author: g.de-boni-rovella
"""

import numpy as np 
import os, time, shutil

#%%------------------------------------------------------------------------
###                            FUNCTIONS
#--------------------------------------------------------------------------

def get_generator_all(n, ebn0_train, batch_size, true_function):

    sigma2 = 1/(2*10**(ebn0_train/10))
    a = 2/sigma2
    sigma2_llr = 4/sigma2
    
    # sigma = llr_max/2
    while True:
        
        # llr random generation
        # llr = sigma*np.random.randn(batch_size,n)
        
        choices = np.random.choice([-a, a], size=(batch_size, n))
        llr = np.random.normal(loc=choices, scale=np.sqrt(sigma2_llr), size=(batch_size, n))
        
        # true LLR of kth position through compact equations
        y_true = np.empty(shape=(batch_size,n))
        for i in range(n):
            y_true[:,i] = true_function(llr,i)
        
        yield (llr, y_true)

def get_Tn(n):
    if n == 2:
        return np.array([[1,0], [1,1]])
    if n == 3:
        return np.array([[1,0,0], [1,1,0], [1,0,1]]) 
    if n == 4:
        return np.array([[1,0,0,0],[1,1,0,0],[1,0,1,0],[1,1,1,1]]) 
    if n == 5:
        return np.array([[1,0,0,0,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0], [1,1,1,0,1]]) 
    if n == 6:
        return np.array([[1,0,0,0,0,0],[1,1,0,0,0,0],[1,0,1,0,0,0],[1,0,0,1,0,0],[1,1,1,0,1,0],[1,1,0,1,0,1]])
    if n == 7:
        return np.array([[1,0,0,0,0,0,0],[1,1,0,0,0,0,0],[1,0,1,0,0,0,0],[1,0,0,1,0,0,0]
                       ,[1,1,1,0,1,0,0],[1,1,0,1,0,1,0],[1,0,1,1,0,0,1]]) 
    if n == 8:
        return np.array([[1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0]
                       ,[1,1,1,0,1,0,0,0],[1,1,0,1,0,1,0,0],[1,0,1,1,0,0,1,0], [1,1,1,1,1,1,1,1]]) 
    if n == 11:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,0,0,0,0],[1,0,0,1,0,0,0,0,0,0,0],[1,1,0,1,1,0,0,0,0,0,0],[0,1,1,1,0,1,0,0,0,0,0]
               ,[1,1,0,0,0,1,1,0,0,0,0],[1,1,0,1,0,0,0,1,0,0,0],[1,0,0,1,1,1,1,0,1,0,0],[1,1,0,0,0,1,0,1,1,1,0],[1,1,1,1,1,0,0,0,1,1,1]])
    if n == 13:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,0,0,0,0,0,0],[1,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,1,0,0,0,0,0,0,0,0]
               ,[1,0,1,0,1,1,0,0,0,0,0,0,0],[0,0,1,1,0,1,1,0,0,0,0,0,0],[0,1,1,1,0,0,0,1,0,0,0,0,0],[1,1,1,0,0,0,0,0,1,0,0,0,0]
               ,[1,1,0,1,0,0,1,1,0,1,0,0,0],[1,1,1,0,0,1,1,0,0,0,1,0,0],[1,1,1,1,0,0,0,0,1,1,1,1,0],[1,1,1,1,1,1,1,1,1,0,0,0,1]])
    if n == 16:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
               ,[1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0],[1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0]
               ,[0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],[1,1,0,1,0,1,1,0,0,0,1,0,0,0,0,0],[1,0,0,1,1,1,0,1,1,0,1,1,0,0,0,0]
               ,[0,1,0,0,1,1,1,1,0,1,1,0,1,0,0,0],[1,1,1,0,0,1,0,0,1,1,1,0,0,1,0,0],[1,0,1,0,0,0,0,1,0,1,1,1,1,0,1,0],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
 
def get_LLR_update(TYPE,T):
    
    n = T.shape[0]
    
    # generate all possible messages
    U_all = np.zeros((2**n,n), dtype=int)
    for i in range(2**n):
        U_all[i] = [int(x) for x in list('{0:0b}'.format(i).zfill(n))]

    # Convert the messages to codewords and BPSK symbols
    C_all = U_all@T % 2

    # generate list to pass to generator
    C_all_0, C_all_1 = [], []
    for i_k in range(n):
        # generate all possible messages that match u_hard in the first i_k-1 bits
        u_hard_0 = np.zeros(i_k+1, dtype=int)
        u_hard_1 = np.append(np.zeros(i_k, dtype=int), np.ones(1, dtype=int), axis=-1)
        
        # indices where the last bit is 0/1
        Ind_0 = np.all(U_all[:,0:i_k+1] == u_hard_0, axis=-1)
        Ind_1 = np.all(U_all[:,0:i_k+1] == u_hard_1, axis=-1)

        C_all_0.append(C_all[Ind_0])
        C_all_1.append(C_all[Ind_1])

    def LLR_update(lc, i_k):
        return np.log( np.sum(np.exp( np.sum( lc[:,None,:] * (1-C_all_0[i_k]), axis=-1) ),axis=-1)/np.sum(np.exp( np.sum( lc[:,None,:] * (1-C_all_1[i_k]), axis=-1) ),axis=-1) )
    
    return LLR_update

#%%------------------------------------------------------------------------
###                            PARAMETERS
#--------------------------------------------------------------------------

# choose the kernel size
n = 16
T = get_Tn(n)

# parameters
batch_size = 2**10
ebn0_train = 5
nb_codewords = 1000 #100*2**n
llr_max = 10 # soft limit for llr max value
data_type = 'zeros' # 'feedback', 'zeros'

# saving parameters
save_path = 'Datasets/T{}_{}-{}dB/'.format(n, data_type,ebn0_train)
for i in range(1,10):
    try:
        os.makedirs(save_path)
        break
    except:
        save_path = save_path[0:-1] + '-/'

shutil.copy(__file__, save_path+'script.py')

#%%------------------------------------------------------------------------
###              Networks and generators definition
#--------------------------------------------------------------------------

# Select the kernel 
LLR_update = get_LLR_update(data_type, T)

# now the generator
train_generator = get_generator_all(n, ebn0_train, batch_size, LLR_update)

#%%------------------------------------------------------------------------
###                 Generate dataset and save them
#--------------------------------------------------------------------------

nb_steps = int(np.ceil(nb_codewords/batch_size))
x = np.empty(shape=(nb_steps*batch_size,n))
y = np.empty(shape=(nb_steps*batch_size,n))
    
time_init = time.time()
for i in range(nb_steps):
    x[i*batch_size:(i+1)*batch_size,:], y[i*batch_size:(i+1)*batch_size,:] = next(iter(train_generator))
        
    t = time.time()-time_init
    # FG.progress(f'Time passed: {int(t)}s',i+1, nb_steps)
    print(f'Time passed: {int(t)}s | remaining (approx): {int( (t/(i+1))*(nb_steps-(i+1)) )}s | {i+1}/{nb_steps}')
print('\n')
np.savez(save_path+f'dataset_T{n}.npz', inputs=x, outputs=y)









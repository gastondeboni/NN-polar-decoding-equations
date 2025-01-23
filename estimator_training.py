#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 23:27:04 2024

@author: gaston
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time, shutil, os

#%%------------------------------------------------------------------------
###                           FUNCTIONS
#--------------------------------------------------------------------------

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

class Tanh(tf.keras.Layer):
    def call(self,x):
        return tf.math.tanh(x)
    
def get_MLP_decoder(n,name,TANH,nb_units):
    in_layer = tf.keras.Input(shape=(n,)) 
    if TANH:
        tanh = Tanh()(in_layer)
        h = tf.keras.layers.Dense(nb_units[0], activation='relu')(tanh)
    else:
        h = tf.keras.layers.Dense(nb_units[0], activation='relu')(in_layer)
    for i in range(1,len(nb_units)):
        h = tf.keras.layers.Dense(nb_units[i], activation='relu')(h)
    out_layer = tf.keras.layers.Dense(1, activation='linear')(h)
    return tf.keras.Model(inputs=in_layer, outputs=out_layer, name=name)

bce = tf.keras.losses.BinaryCrossentropy()
def KL_divergence_loss(y_true,y_pred): 
    p_true = tf.math.sigmoid(y_true/1.0)
    p_pred = tf.math.sigmoid(y_pred/1.0)
    return bce(p_true,p_pred) - bce(p_true,p_true)

def get_LLR_update(TYPE,T):
    
    n = T.shape[0]
    if TYPE == 'zeros':
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

# to measure progress during Bit Error Rate testing
def progress(message, current, total, bar_length=30):
    fraction = min(current / total,1)
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current >= total else '\r'
    print(f'\r{message}: [{arrow}{padding}] - {current}/{total}', end=ending)   

#%%------------------------------------------------------------------------
###                            PARAMETERS
#--------------------------------------------------------------------------

# choose the kernel size
n = 16

# training parameters
batch_size = 2**10
epochs = 2000
patience = 20 # in epochs
ebn0_train = [1,3,5]
learning_rate = 5e-4
VERBOSE = 0
TANH = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# testing parameters
test_N = 20
min_errors = 300
nb_sim_max = 5000
ebn0_test = np.arange(0,8,1)
test_batch_size = 2**10

# saving parameters 
save_path = f'Models_T{n}_zeros_{sum(TANH)}tanh_{ebn0_train[0]}-{ebn0_train[-1]}dB/'
os.makedirs(save_path)
os.makedirs(save_path+'models/')
os.makedirs(save_path+'summaries/')
os.makedirs(save_path+'training_metrics/')

# save a copy of important files
shutil.copy(__file__, save_path+'script.py')

#%%------------------------------------------------------------------------
###              Training data retreival
#--------------------------------------------------------------------------

# Select the kernel
T = get_Tn(n)

x, y = [], []
for i_ebn0 in range(len(ebn0_train)):
    data = np.load(f'Datasets/T{n}_zeros-{ebn0_train[i_ebn0]}dB/dataset_T{n}.npz')
    x.append(data['inputs'])
    y.append(data['outputs'])
    
x, y = np.concatenate(x), np.concatenate(y)
permutation = np.random.permutation(x.shape[0])
x, y = x[permutation], y[permutation]

#%%------------------------------------------------------------------------
###                        Training
#--------------------------------------------------------------------------

loss_list, epochs_list = [], []
u = [6,12,8,8,16,12,12,12,12,12,8,4,4,2,1,1]
for i in range(n):
    if u[i] == 0:
        continue
    
    # choose model and compile
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model =  get_MLP_decoder(n,f'model_{i}',TANH=TANH[i],nb_units=[u[i]*n,u[i]*int(n/2),u[i]*int(n/4)])
    model.compile(optimizer=optim, loss=KL_divergence_loss)
    
    model.summary() 
    params = model.count_params()
    with open(save_path+f'summaries/modelsummary_{i}_{params//1000}k.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # define callbackas and train
    EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss',mode='min', patience=patience, verbose=0)
    init_train = time.time()
    history = model.fit(x,y[:,i],batch_size,epochs=epochs,callbacks=[EarlyStop], verbose=VERBOSE)
    train_time = time.time() - init_train
    print('\nTraining time = {}h {}m {}s\n'.format(int(train_time//3600),int(train_time%3600)//60, int(train_time%3600) % 60))
    
    # save model
    model.save(save_path+'models/estimator_T{}_{}.h5'.format(n,i), save_format='h5', include_optimizer=False)

    # plot learning curves
    loss = history.history['loss'][:]
    epochs_vec = np.arange(len(loss))
    loss_list.append(loss)
    epochs_list.append(epochs_vec)

    plt.figure()
    plt.semilogy(epochs_vec, loss, '-b')
    plt.title('Training loss')
    plt.grid('on', which='both', ls='--')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(save_path+'training_metrics/training_metrics_{}.eps'.format(i), format='eps', dpi=600)


plt.figure()
for i in range(n):
    linetype = (i<=n/2)*'-' + (i>n/2)*':'
    plt.semilogy(epochs_list[i], loss_list[i], linetype, label=f'{i}')
plt.title('Training loss')
plt.grid('on', which='both', ls='--')
plt.xlabel('epoch')
plt.legend()
plt.savefig(save_path+'loss_all.eps', format='eps', dpi=600)

#%% 

L_estimators = []
for i in range(n):
    L_estimators.append(tf.keras.models.load_model(save_path + f'models/estimator_T{n}_{i}.h5', compile=False,
                                                   custom_objects={'Tanh':Tanh}))

    params = L_estimators[-1].count_params()
    with open(save_path+f'summaries/modelsummary_{i}_{params//1000}k.txt', 'w') as f:
        L_estimators[-1].summary(print_fn=lambda x: f.write(x + '\n'))

#%%------------------------------------------------------------------------
###                        Testing
#--------------------------------------------------------------------------

LLR_update = get_LLR_update('zeros',T)
metric_estimator = np.zeros((n,ebn0_test.shape[0]))
sign_errors = np.zeros((n,ebn0_test.shape[0]))

for i_ebn0 in range(ebn0_test.shape[0]):
    print(f'Testing Eb/N0 = {ebn0_test[i_ebn0]}')
    sigma2 = 1/(2*10**(ebn0_test[i_ebn0]/10))
    a = 2/sigma2
    sigma2_llr = 4/sigma2
    
    for i in range(n):
        metric, signs = 0, 0
        for j in range(test_N):
            progress(f'Estimator {i}: ',j+1, test_N)
            
            choices = np.random.choice([-a, a], size=(test_batch_size, n))
            llr = np.random.normal(loc=choices, scale=np.sqrt(sigma2_llr), size=(test_batch_size, n))
            
            # true LLR of kth position through marginalization
            y_true = LLR_update(llr,i).astype('float32')
            
            # estimated L updating - supposing previous bits are correct
            y_est = L_estimators[i](llr)[:,0].numpy()
            
            # compute errors
            metric += np.mean( KL_divergence_loss(y_true, y_est) )
            signs += np.mean( np.sign(y_true) != np.sign(y_est) )
        
        metric_estimator[i,i_ebn0] = metric/test_N
        sign_errors[i,i_ebn0] = signs/test_N
        
    
plt.figure()
for i in range(n):
    linetype = (i<=n/2)*'-' + (i>n/2)*':'
    plt.semilogy(ebn0_test, metric_estimator[i], linetype, label=f'{i}')
plt.title('KL for each estimator')
plt.grid('on', which='both', ls='--')
plt.xlabel('Eb/N0')
plt.legend(ncol=4, loc='lower left')
plt.savefig(save_path+'KL_each.eps', format='eps', dpi=600)

plt.figure()
for i in range(n):
    linetype = (i<=n/2)*'-' + (i>n/2)*':'
    plt.plot(ebn0_test, 100*sign_errors[i], linetype)
plt.title('percentage of sign errors')
plt.grid('on', which='both', ls='--')
plt.xlabel('Eb/N0')
plt.ylabel('% sign errors')
plt.legend(list(range(n)), ncol=4, loc='best')
plt.savefig(save_path+'sign_errors_each_all.eps', format='eps', dpi=600)
plt.ylim(top=5, bottom=0)
plt.savefig(save_path+'sign_errors_each.eps', format='eps', dpi=600)


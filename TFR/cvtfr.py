#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit Tensor Factored Regression model
Cross-validation

Created on Thu Dec 13 2019

@author: jmagraner
"""

#%% SLURM custer scheduler specs

# As we are submitting a large number of jobs, add these lines before importing
# anything to enforce SLURM resource allocation, since setting the SLURM flag 
# cpus-per-task=1 only informs SLURM of the task needs

import os                                                                                                                                                                                                                                       
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"                                                                                                                                                          
os.environ["MKL_NUM_THREADS"] = "1"                                                                                                                                                      
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"                                                                                                                                                     
os.environ["NUMEXPR_NUM_THREADS"] = "1"     


#%% Import packages and functions

import autograd.numpy as np
import scipy.io as spio
import time
import pickle
import os.path

from sys import argv

from core.optimizetfr import fitmodel
from core.utiltfr     import apply_orth_transform
from core.outputtfr   import generate_output
from pathstfr         import mypaths


# %%
#### Specs ####

monkey   = 'ar'
model    = 'TFR'
modeldir = 'ABcx_2Dinp'

## model dimensionality
xdim     = int(argv[1])  # latent subspace dimensionality
udim     = 4             # input subspace dimensionality

## across contexts parameter constraints
ABfixed    = False
Cfixed     = True
Ufixed     = True
Tiofixed   = True
cohfixed   = True
biasfixed  = False
biastfixed = False

## input time courses constraints
inputs_u           = False   # learn full input time courses
inputs_coh_TinTout = True    # learn constrained input time courses
inputs_coh         = False   # learn scalar inputs, constant in time

## initialization
randinit  = True   # default, if nothing specified
otherinit = False  # alternatively, implement your own init

## input regularization
inputreg  = False

## orthogonal transform
orthtransform = True


## paths
modelpath, datapath = mypaths()
# saved results
savepath  = (modelpath + monkey + '/' + model + '/cross-validation/' + 
             modeldir + '/')
savename  = (monkey + model + '_' + modeldir + '_h' + str(int(argv[1])) +
             'cv' + str(int(argv[2])))

## data
# Y(neurons,time,motion,color,context)
# color&motion:  1==-0.5,    6==0.5
# context:       1==color,   2==motion

data = 'Y_meanfiringrates_Zscored_nonsmooth.mat'
mat  = spio.loadmat(datapath + monkey +'/' + data, squeeze_me=True)
Y    = mat['Y']
data_dims = Y.shape
ndim,tdim,mdim,cdim,cxdim = data_dims
kdim      = cdim*mdim  #conditions

data_dims = Y.shape
ndim,tdim,mdim,cdim,cxdim = data_dims
kdim      = cdim*mdim  #conditions

tiodim    = 2        # Tin-Tout type of evidence: # of distinct time courses
cohdim    = np.maximum(mdim,cdim) 
lat_dims  = (udim, xdim, cohdim, tiodim)

## algorithmic
max_iter  = 10000    # max iterations
min_iter  = 5000     # min iterations
tol       = 1e-5     # stopping criteria MSE(t)-MSE(t-1) < tol
step_size = 0.009    # a bit more than ADAM default 0.01, prone to local optima
MSE       = np.inf   # init loss


# %% 
#### Define model target output (the PFC data) and indicator matrices ####

# indicator matrices
Q  = np.zeros((udim+1,udim*cohdim+1,udim*tiodim+1,kdim))  # [for mot/col, in condition, which coherence?]
Qu = np.zeros((2,cdim,tiodim))  # [for mot/col, in coherence, is Tin or Tout? ]

# experimental coherences (only used to help grup conditions)
cond_mot=np.array([-0.50, -0.15, -0.05, 0.05, 0.15, 0.50])
cond_col=np.array([-0.50, -0.18, -0.06, 0.06, 0.18, 0.50])

Tin_motk  = np.repeat(np.diag(cond_col),mdim,axis=1)>0
Tout_motk = np.repeat(np.diag(cond_col),mdim,axis=1)<0
Tin_colk  = np.tile(np.diag(cond_mot),cdim)>0
Tout_colk = np.tile(np.diag(cond_mot),cdim)<0

Q[0,0,0,:] = np.ones((1, kdim))

count = 1
for n in range(1,udim+1):
    init = 1+mdim*(n-1)
    nn   = range(init,init+mdim)
    if np.remainder(n,2)==0:
        Q[n,nn,count,:]   = Tout_colk
        Q[n,nn,count+1,:] = Tin_colk
    else:
        Q[n,nn,count,:]   = Tout_motk
        Q[n,nn,count+1,:] = Tin_motk
    count = count+2
        
# LDS-like cummulative time indicator (to implement temporal convolution)
P = np.zeros((tdim,tdim,tdim))
for t in range(0,tdim):
    for tt in range(0,tdim):
        ttt = t-tt
        if ttt>=0:
            P[t,tt,ttt]=1    
            

# target
target = []

for cx in range(cxdim):
    Ycx    = np.zeros((ndim,tdim,kdim))
    k  = -1
    for i in range(mdim):
        for j in range(cdim):
            k = k+1
            Ycx[:,:,k] = Y[:,:,i,j,cx]
            
            if i<=2: Qu[0,i,0] = 1
            if j<=2: Qu[1,j,0] = 1 
            if i>2 : Qu[0,i,1] = 1
            if j>2 : Qu[1,j,1] = 1

    target.append(Ycx)
  
Qu = np.tile(Qu,(int(udim/2),1,1))


### Cross-validation ###

# remove condition
k = int(argv[2])-1 #careful, python indexing!
targetk = []
for cx in range(cxdim):
    targetk.append(target[cx][:,:,k].reshape(ndim,tdim,1))
    target[cx] = np.delete(target[cx],k,axis=2)
    
Qk = Q[:,:,:,k].reshape(udim+1,udim*cohdim+1,udim*tiodim+1,1)    
Q  = np.delete(Q,k,axis=3)
kdim = kdim - 1


cxconst  = ABfixed, Cfixed, Ufixed, Tiofixed, cohfixed, biasfixed, biastfixed
inpconst = inputs_u, inputs_coh_TinTout, inputs_coh

specs = {
'monkey'      : monkey,
'model'       : model,
'modeldir'    : modeldir,   
'data file'   : data,       
'data dims'   : data_dims,
'conditions'  : kdim,
'latent dims' : lat_dims,       
'contextual constraints' : cxconst,
'input constraints'      : inpconst,
'input regularization'   : inputreg,
'max iter' : max_iter,
'min iter' : min_iter,
'iter'     : max_iter,
'step size': step_size,
'tol': tol,
'MSE': [MSE],
'P'  : P,
'Q'  : Q,
'Qk' : Qk,
'Qu' : Qu,
}


# %% 
#### Initialize model ####

LDSparams = []

if otherinit:
    # your initialization 
    from core.inittfr import initparams_myinit as initparams
    initype = 'myinit'
    init_params = initparams(specs,LDSparams,initype)   
else:
    from core.inittfr import initparams_random as initparams
    initype = []
        

### Initialize parameters ###    
init_params = initparams(specs,LDSparams,initype)


# %%
### Compute gradients and perform optimization###

print("Fitting TFR model...")
print()  

start_time = time.time()

# Global variables are far from ideal... but this worked well
global trained_params

# fit model
trained_params,specs = fitmodel(init_params,target,specs)


# input time courses
if inputs_coh_TinTout:
  for cx in range(cxdim):
    coh = trained_params['coh'][0]  if cohfixed  else  trained_params['coh'][cx]  
    Tio = trained_params['Tio'][0]  if Tiofixed  else  trained_params['Tio'][cx]
    Uio = np.zeros((udim,cdim,tdim))
    for u in range(udim):
      Uio[u,:,:] = coh[u,:].reshape(1,-1).T * Qu[u,:,:] @ Tio[u,:,:]
    if Ufixed:
        trained_params['U'][0] = Uio 
    else:
        trained_params['U'][cx] = Uio 
    
if inputs_coh:
  for cx in range(cxdim):
    if Tiofixed:
        trained_params['Tio'][0]  = np.ones((udim,tiodim,tdim))
    else:
        trained_params['Tio'][cx] = np.ones((udim,tiodim,tdim))  
    coh = trained_params['coh'][0]  if cohfixed  else  trained_params['coh'][cx]
    Tio = trained_params['Tio'][0]  if Tiofixed  else  trained_params['Tio'][cx]
    Uio = np.zeros((udim,cdim,tdim))
    for u in range(udim):
      Uio[u,:,:] = coh[u,:].reshape(1,-1).T * Qu[u,:,:] @ Tio[u,:,:]
    if Ufixed:
        trained_params['U'][0] = Uio 
    else:
        trained_params['U'][cx] = Uio 
      
    
# orthogonalise output matrix C post-hoc
if orthtransform:    
    trained_params = apply_orth_transform(trained_params,specs)

# generate model predictions for left-out condition
specs['Q'] = Qk
specs['conditions'] = 1

Ypred, MSE  = generate_output(trained_params,targetk,specs)

specs['Q'] = Q
specs['conditions'] = kdim

print()
print('CV MSE color context = ', MSE[0], ',  CV MSE motion context = ', MSE[1])

# saving otput
if not os.path.exists(savepath):
    os.makedirs(savepath)
        
with open(savepath+savename, 'wb') as f:
    pickle.dump([init_params,trained_params, specs, Ypred, MSE], f)
    

print()  
print("--- elapsed time:  %s seconds ---" % (time.time() - start_time))
print()  


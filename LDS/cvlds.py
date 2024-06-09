#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit non-probabilistic LDS model
Cross-validation

Created on Thu Jun 13 2019

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

from core.optimizemodel import fitmodel
from core.utilities     import apply_orth_transform
from core.modeloutput   import generate_output
from mypaths            import mypaths


# %% 
#### Specs ####

monkey   = 'ar'
model    = 'LDS'
modeldir = 'ABcx_3Dinp'


## model dimensionality
xdim     = int(argv[1])  # latent subspace dimensionality
udim     = 6             # input subspace dimensionality (mot + col)

## across contexts parameter constraints
Afixed   = True
Bfixed   = False
Cfixed   = True
dfixed   = True
x0fixed  = False
Ufixed   = True
Tiofixed = True
cohfixed = True

## input time courses constraints
inputs_u           = False   # learn full input time courses
inputs_coh_TinTout = True    # learn constrained input time courses
inputs_coh         = False   # learn scalar inputs, constant in time

## initialization
randinit  = True   # default, if nothing specified
otherinit = False  # alternatively, implement your own init

## input regularization
inputreg  = True
# Lagrange multiplier
inplm     = 1e-5 # Default to 1e-5


# Constrain the dynamics matrix to be approximately normal (AA^T = A^TA)
Anormal   = False
# Lagrange multipliers
anormlm   = 1e5  # Default to 1e5 (strongly enforce normality)
orthlm    = 0    # Default to 1e5 (strongly enforce orthogonality of C)
# It does not work as intended, better set C to a prevously learned
# orthogonalized matrix (post-hoc) and mask the gradient for C.

## orthogonal transform
orthtransform = True

if Anormal:
    orthtransform = False # Do not apply this if Corth is already enforced

## train subset of parameters only. To specify which, SET mask in 
#  core/initmodel.py, function gradmask
train_subset = False
# Init model. The unchanged parameters will be set to the ones of this 
# pre-trained model
initLDSmodel = 'ABcx_3Dinp'
# Initialize at random the parameters to be learned or initialize them with the
# previous model values. The later might get the optimization stuck in a local
# optimum, so set this to True. MUST specify WHICH parameters manually BELOW!
# This SHOULD match the parameters set to 1 in the mask in core/initmodel.py
set_to_random = True
# Note that to implement this functionality we had to modify the core autograd 
# adam function within optimizers.py, to mask the gradients.

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

tiodim    = 2          # Tin-Tout type of evidence: # of distinct time courses
cohdim    = np.maximum(mdim,cdim) 
lat_dims  = (udim, xdim, cohdim, tiodim)

## algorithmic
max_iter  = 10000    # max iterations
min_iter  = 5000     # min iterations (helps skeeping local optima)
tol       = 1e-5     # stopping criteria MSE(t)-MSE(t-1) < tol
step_size = 0.009    # a bit less than ADAM default 0.01, prone to local optima
MSE       = np.inf   # init loss


# %% 
#### Define model target output (the PFC data) and indicator matrices ####

# indicator matrices
Q  = np.zeros((2,kdim,cdim))    # [for mot/col, in condition, which coherence?]
Qu = np.zeros((2,cdim,tiodim))  # [for mot/col, in coherence, is Tin or Tout? ]

target = []

for cx in range(cxdim):
  Ycx    = np.zeros((ndim,kdim,tdim))

  k  = -1
  for i in range(mdim):
    for j in range(cdim):
      k = k+1

      Ycx[:,k,:]  = Y[:,:,i,j,cx]
      
      Q[0,k,i] = 1
      Q[1,k,j] = 1
      
      if i<=2: Qu[0,i,0] = 1
      if j<=2: Qu[1,j,0] = 1 
      if i>2 : Qu[0,i,1] = 1
      if j>2 : Qu[1,j,1] = 1

  target.append(Ycx)


### Cross-validation ###

# remove condition
k = int(argv[2])-1 #careful, python indexing!
targetk = []
for cx in range(cxdim):
    targetk.append(target[cx][:,k,:].reshape(ndim,1,tdim))
    target[cx] = np.delete(target[cx],k,axis=1)
    
Qk = Q[:,k,:].reshape(2,1,cdim)    
Q  = np.delete(Q,k,axis=1)
kdim = kdim - 1

Q  = np.tile(Q,(int(udim/2),1,1))
Qk = np.tile(Qk,(int(udim/2),1,1))
Qu = np.tile(Qu,(int(udim/2),1,1))

cxconst  = Afixed, Bfixed, Cfixed, dfixed, x0fixed, Ufixed, Tiofixed, cohfixed
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
'input constraints': inpconst,
'input regularization': inputreg,
'normal dynamics penalty': Anormal,
'input regularization lagrange multiplier': inplm,
'normal dynamics lagrange multiplier': anormlm,
'orth constraint lagrange multiplier': orthlm,
'train subset' : train_subset,
'max iter' : max_iter,
'min iter' : min_iter,
'iter' : max_iter,
'step size': step_size,
'tol': tol,
'MSE': [MSE],
'Q' : Q,
'Qk': Qk,
'Qu': Qu,
}


# %% 
#### Initialize model ####

LDSparams = []

if otherinit:
    # your initialization 
    from core.initmodel import initparams_myinit as initparams
    initype = 'myinit'
    init_params = initparams(specs,LDSparams,initype)
else:
    # random initialization        
    from core.initmodel import initparams_random as initparams
    initype   = []
        
if train_subset:
    # Load model to re-learned some of its parameters
    LDSpath  = modelpath+monkey+'/LDS'+'/cross-validation/'+initLDSmodel+'/'
    LDSfile  = (monkey + 'LDS_' + initLDSmodel + '_h' + str(int(argv[1])) + 
                'cv' + str(int(argv[2])))
    with open(LDSpath+LDSfile, 'rb') as f:
        _,init_params,_,_,_    = pickle.load(f)
    if set_to_random:
        # Initialize the re-learned parameters at random
        init_params_random = initparams(specs,LDSparams,initype)
        init_params['B']   = init_params_random['B']
        init_params['U']   = init_params_random['U']
        init_params['Tio'] = init_params_random['Tio']
        init_params['coh'] = init_params_random['coh']    
else:
    if randinit:
        # Initialize all parameters at random
        init_params = initparams(specs,LDSparams,initype)


# %% 
#### Compute gradients and perform optimization ####
print()
print("Fitting non-probabilistic LDS...")
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

Ypred, MSE = generate_output(trained_params,targetk,specs)

specs['Q'] = Q
specs['conditions'] = kdim

print()
print('CV MSE color context = ', MSE[0], ',  CV MSE motion context = ', MSE[1])

# saving output
if not os.path.exists(savepath):
    os.makedirs(savepath)
        
with open(savepath+savename, 'wb') as f:
    pickle.dump([init_params, trained_params, specs, Ypred, MSE], f)
    

print()  
print("--- elapsed time:  %s seconds ---" % (time.time() - start_time))
print()  


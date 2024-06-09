#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 2019

@author: jmagraner
"""

# %%
from autograd import grad
from autograd.misc.optimizers import adam
import autograd.numpy as np


def fitmodel(params,target,specs):
    
    
    ### Define LDS as a recurrent neural network ###   
    def LDSmodel(params):
    
        def update_hiddens(input, hiddens):
          
            hidd_rec_ac = np.dot(A, hiddens)
            hidd_inp_ac = np.dot(B, input)
            
            return hidd_rec_ac + hidd_inp_ac
    
        def hiddens_to_output(hiddens):    
            output = np.dot(C, hiddens) + d
            return output
          
          
        def compute_inputs_u(U):      
            inputs = np.array([]).reshape(0,kdim,tdim)
            for u in range(udim):
              inp    = Q[u,:,:] @ U[u,:,:]
              inputs = np.concatenate((inputs,inp[None,...]),axis=0)          
            return inputs
        
        def compute_inputs_coh_TinTout(Tio,coh):      
            inputs = np.array([]).reshape(0,kdim,tdim)
            for u in range(udim):
              Uio    = coh[u,:].reshape(1,-1).T * Qu[u,:,:] @ Tio[u,:,:] 
              inp    = Q[u,:,:] @ Uio
              inputs = np.concatenate((inputs,inp[None,...]),axis=0)          
            return inputs
        
        def compute_inputs_coh(coh):      
            inputs = np.array([]).reshape(0,kdim,tdim)
            Tio    = np.ones((udim,tiodim,tdim))
            for u in range(udim):
              Uio    = coh[u,:].reshape(1,-1).T * Qu[u,:,:] @ Tio[u,:,:] 
              inp    = Q[u,:,:] @ Uio
              inputs = np.concatenate((inputs,inp[None,...]),axis=0)          
            return inputs
          
          
        hiddens_stack_cx = []
        output_stack_cx  = []
        inputs_stack_cx  = []
        
        for cx in range(cxdim):      
          A   = params['A'][0]    if Afixed    else  params['A'][cx]
          B   = params['B'][0]    if Bfixed    else  params['B'][cx] 
          C   = params['C'][0]    if Cfixed    else  params['C'][cx]
          d   = params['d'][0]    if dfixed    else  params['d'][cx]
          x0  = params['x0'][0]   if x0fixed   else  params['x0'][cx]
          U   = params['U'][0]    if Ufixed    else  params['U'][cx]
          Tio = params['Tio'][0]  if Tiofixed  else  params['Tio'][cx]
          coh = params['coh'][0]  if cohfixed  else  params['coh'][cx]
    
          if inputs_u:
            inputs = compute_inputs_u(U)
            
          if inputs_coh_TinTout:
            inputs = compute_inputs_coh_TinTout(Tio,coh)
            
          if inputs_coh:
            inputs = compute_inputs_coh(coh)
          
          hiddens = x0
    
          hiddens_stack = np.array([]).reshape(xdim,kdim,0)
          output_stack  = np.array([]).reshape(ndim,kdim,0)
    
          # Iterate over time steps
          for input in inputs.T:
            hiddens = update_hiddens(input.T, hiddens)
            output  = hiddens_to_output(hiddens)
    
            hiddens_stack = np.concatenate((hiddens_stack,
                                            hiddens[...,None]),axis=2)
            output_stack  = np.concatenate((output_stack,
                                            output[...,None]),axis=2)
          
          hiddens_stack_cx.append(hiddens_stack)
          output_stack_cx.append(output_stack)
          inputs_stack_cx.append(inputs)

      
        return output_stack_cx, hiddens_stack_cx, inputs_stack_cx
      
      
    ### Input squared norm penalty ###    
    def LDSinpnorm(params,inp,cx):
        B   = params['B'][0]    if Bfixed    else  params['B'][cx]
        inpnorm = 0
        for k in range(kdim):
            for t in range(tdim):
                Bu = np.dot(B, inp[:,k,t])
                inpnorm = inpnorm + np.dot(Bu,Bu)
        return inpnorm

    ### Dynamics matrix normal constraint ###
    def LDSAnormal(params,cx):
        A     = params['A'][0]    if Afixed    else  params['A'][cx]
        Acost = A @ A.T - A.T @ A
        Acostnorm = np.trace(Acost @ Acost.T)
        # Must orthogonalize C as well for this to work
        C     = params['C'][0]    if Cfixed    else  params['C'][cx]
        Ccost = C.T @ C - np.eye(C.shape[1])
        Ccostnorm = np.trace(Ccost @ Ccost.T)
        return Acostnorm, Ccostnorm
        
    ### Define loss function ###    
    def training_loss(params,iter):    
        output,_,inputstack = LDSmodel(params)
        mse     = 0
        inpnorm = 0
        normalp = 0
        orthonp = 0
        for cx in range(cxdim):
          if inputreg:  
              inpnorm = LDSinpnorm(params,inputstack[cx],cx)
          if Anormal:
              normalp, orthonp = LDSAnormal(params,cx)
          mse = mse + np.mean((output[cx] - target[cx])**2)/cxdim + ilm*inpnorm + nlm*normalp + olm*orthonp
        return mse
    

    ### Callback function for stopping criteria ###    
    def callback(params, iter, gradient):
        mse_old = specs['MSE'][-1]
        mse_new = training_loss(params, 0)    
        tol     = mse_old - mse_new
        specs['MSE'].append(mse_new)
        print('iteration', iter, ': MSE = ', mse_new)
        if 0<tol<specs['tol'] and iter>specs['min iter']:
            global trained_params
            trained_params = params
            specs['iter']  = iter
            print()
            print('Terminating... convergence criteria met')
            raise Exception
    
    
    #### 
    # model specifications
    ndim, tdim, mdim, cdim, cxdim  = specs['data dims']
    udim, xdim, cohdim, tiodim     = specs['latent dims']
    
    kdim = specs['conditions']
    
    Afixed,Bfixed,Cfixed,dfixed ,_,_,_,_      = specs['contextual constraints']
    _,_,_,_, x0fixed,Ufixed,Tiofixed,cohfixed = specs['contextual constraints']

    inputs_u, inputs_coh_TinTout, inputs_coh  = specs['input constraints']

    Q =  specs['Q']
    Qu = specs['Qu']

    step_size = specs['step size']
    max_iter  = specs['max iter']
    
    inputreg  = specs['input regularization']
    Anormal   = specs['normal dynamics penalty']
    ilm       = specs['input regularization lagrange multiplier']
    nlm       = specs['normal dynamics lagrange multiplier']
    olm       = specs['orth constraint lagrange multiplier']


    # Build gradient of loss function using autograd.
    training_loss_grad = grad(training_loss)
    
    # Load mask to zero-out gradients for params that we don't want to change.
    # Note that we modified the core autograd function adam within the script
    # optimizers.py to implement this, by adding the flag flatten_mask.
    if specs['train subset']:
        from core.initmodel import gradmask
        from autograd.misc import flatten
        grad_mask    = gradmask(specs)
        flatten_obj  = flatten(grad_mask)
        flatten_mask = flatten_obj[0].astype(bool)
    else:
        flatten_mask = False
    
    
    try:
        # Learn parameters using adam optimizer
        params = adam(training_loss_grad, params, step_size=step_size,
                      num_iters=max_iter, callback=callback, 
                      grad_mask=flatten_mask)
        global trained_params
        trained_params = params
        pass
    except:
        return trained_params, specs
              
    return trained_params,specs
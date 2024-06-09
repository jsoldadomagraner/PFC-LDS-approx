#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 2019

@author: jmagraner
"""

# %%
import autograd.numpy as np


def generate_output(params,target,specs):
    
    
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
      
      

    ### Define loss function ###    
    def training_loss(params):    
        output,_,_ = LDSmodel(params)
        MSE = []
        for cx in range(cxdim):
          mse     = np.mean((output[cx] - target[cx])**2)
          MSE.append(mse)
        return output, MSE
    
    
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

    output, mse  = training_loss(params)

    
    return  output, mse
  
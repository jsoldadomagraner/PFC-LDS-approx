#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 2019

@author: jmagraner
"""

# %%
import autograd.numpy as np


def generate_output(params,target,specs):
    
    
    ### Define tensor product ###   
    def TFRmodel(params):
        
        # C (nh)    , AB (ht''l) 
        # QU (lkt') , E (tt't'')
        def compute_outputs(QU):
            CAB = np.tensordot(C,AB,axes=(1,0))                # nt''l
            QUP = np.tensordot(QU,P, axes=(2,1))               # lktt''
            QUP = np.moveaxis(QUP,[0,1,3],[1,-1,0])            # t''ltk
            Y   = np.tensordot(CAB, QUP, axes=([1,2], [0,1]))  # ntk
            return Y
          
        # Q (lmk), U (mt)    
        def compute_inputs_u(U):   
            # Q (lmik) remove third dim
            QU  = np.tensordot(Q, U, axes=(1,0));  # lkt'
            return QU
            
        # Q (lmik), coh (m), Tio (it')
        def compute_inputs_coh_TinTout(Tio,coh):
            Qcoh = np.tensordot(Q, coh, axes=(1, 0))     # lik
            QU   = np.tensordot(Qcoh, Tio, axes=(1,0))   # lkt'
            return QU
        
        def compute_inputs_coh(coh):
            Tio  = np.ones((udim*tiodim+1,tdim))
            Qcoh = np.tensordot(Q, coh, axes=(1, 0))     # lik
            QU   = np.tensordot(Qcoh, Tio, axes=(1,0))   # lkt'
            return QU
          
        Ycx  = []
        QUcx = []
        
        for cx in range(cxdim):      
          AB   = params['AB'][0]    if ABfixed    else  params['AB'][cx]
          C    = params['C'][0]     if Cfixed     else  params['C'][cx]
          U    = params['U'][0]     if Ufixed     else  params['U'][cx]
          Tio  = params['Tio'][0]   if Tiofixed   else  params['Tio'][cx]
          coh  = params['coh'][0]   if cohfixed   else  params['coh'][cx]
          b    = params['bias'][0]  if biasfixed  else  params['bias'][cx]
          bt   = params['biast'][0] if biastfixed else  params['biast'][cx]

          U   = np.reshape(U,(udim*cohdim,tdim))
          Tio = np.reshape(Tio,(udim*tiodim,tdim))
          coh = np.reshape(coh,(udim*cohdim))
          U   = np.concatenate((bt,U))
          Tio = np.concatenate((bt,Tio))
          coh = np.concatenate((b,coh))
          
          if inputs_u:
              QU = compute_inputs_u(U)
            
          if inputs_coh_TinTout:
              QU = compute_inputs_coh_TinTout(Tio,coh)
            
          if inputs_coh:
              QU = compute_inputs_coh(coh)
          
          Y = compute_outputs(QU)
          
          Ycx.append(Y)
          QUcx.append(QU)
                
        return Ycx, QUcx
      
            
    ### Define loss function ###    
    def training_loss(params):    
        output,_ = TFRmodel(params)
        MSE = []
        for cx in range(cxdim):
          mse = np.mean((output[cx] - target[cx])**2)
          MSE.append(mse)
        return output, MSE   
    
    
    #### 
    # model specifications
    ndim, tdim, mdim, cdim, cxdim  = specs['data dims']
    udim, xdim, cohdim, tiodim     = specs['latent dims']
        
    ABfixed,Cfixed,Ufixed,Tiofixed,_,_,_  = specs['contextual constraints']
    _,_,_,_,cohfixed,biasfixed,biastfixed = specs['contextual constraints']

    inputs_u, inputs_coh_TinTout, inputs_coh = specs['input constraints']

    P = specs['P']
    Q = specs['Q']

    output, mse  = training_loss(params)

              
    return output, mse
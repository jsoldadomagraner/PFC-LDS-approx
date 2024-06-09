#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions

Created on Thu Jun 13 2019

@author: jmagraner
"""

# %%
import autograd.numpy as np


### Orthogonalise output matrix ###

def apply_orth_transform(params,specs):
  
# NOTE: more careful implementation needed! 
# This implementation does not support having different output matrices C 
# across contexts together with other parameters. 
# We are assuming here that C is always fixed across contexts.

    ndim, tdim, mdim, cdim, cxdim  = specs['data dims']
    udim, xdim, cohdim, tiodim     = specs['latent dims']
    
    Afixed,Bfixed,Cfixed,dfixed ,_,_,_,_      = specs['contextual constraints']
    _,_,_,_, x0fixed,Ufixed,Tiofixed,cohfixed = specs['contextual constraints']
    
    for cx in range(cxdim): 
      
      A  = params['A'][0]  if Afixed  else  params['A'][cx]
      B  = params['B'][0]  if Bfixed  else  params['B'][cx] 
      C  = params['C'][0]  if Cfixed  else  params['C'][cx]
      x0 = params['x0'][0] if x0fixed else  params['x0'][cx]
  
      _,SC,VCh = np.linalg.svd(C,full_matrices=False)
          
      M     = np.diag(SC) @ VCh
      Minv  = VCh.T @ np.linalg.inv(np.diag(SC))
                 
      Anew  = M @ A @ Minv
      Bnew  = M @ B
      Cnew  = C @ Minv
      x0new = M @ x0      

      if not Afixed:  params['A'][cx]  = Anew   
      if not Bfixed:  params['B'][cx]  = Bnew
      if not Cfixed:  params['C'][cx]  = Cnew
      if not x0fixed: params['x0'][cx] = x0new        
        
    if Afixed:  params['A'][0]  = Anew
    if Bfixed:  params['B'][0]  = Bnew
    if Cfixed:  params['C'][0]  = Cnew
    if x0fixed: params['x0'][0] = x0new
    
            
    return params
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
    
    ABfixed,Cfixed,Ufixed,Tiofixed,_,_,_  = specs['contextual constraints']
    _,_,_,_,cohfixed,biasfixed,biastfixed = specs['contextual constraints']
    
    for cx in range(cxdim): 
      
      AB = params['AB'][0] if ABfixed  else  params['AB'][cx]
      C  = params['C'][0]  if Cfixed   else  params['C'][cx]
  
      _,SC,VCh = np.linalg.svd(C,full_matrices=False)
          
      M     = np.diag(SC) @ VCh
      Minv  = VCh.T @ np.linalg.inv(np.diag(SC))
                 
      ABnew = np.tensordot(M,AB,axes=(1,0))
      Cnew  = C @ Minv      

      if not ABfixed: params['AB'][cx] = ABnew   
      if not Cfixed:  params['C'][cx]  = Cnew
        
    if ABfixed: params['AB'][0] = ABnew
    if Cfixed:  params['C'][0]  = Cnew
    
            
    return params
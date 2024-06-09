#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize LDS model

Created on Thu Jun 13 2019

@author: jmagraner
"""

# %%
import autograd.numpy as np
import autograd.numpy.random as npr


def initparams_random(specs,*unused):
  
    sc=0.01
    
    ndim, tdim, mdim, cdim, cxdim  = specs['data dims']
    udim, xdim, cohdim, tiodim     = specs['latent dims']
    
    Afixed,Bfixed,Cfixed,dfixed ,_,_,_,_      = specs['contextual constraints']
    _,_,_,_, x0fixed,Ufixed,Tiofixed,cohfixed = specs['contextual constraints']
    
    A_L   = []
    B_L   = []
    C_L   = []
    d_L   = []
    x0_L  = []
    U_L   = []
    Tio_L = []
    coh_L = []
    
    cx2  = True
    
    for cx in range(cxdim):
    
      A    = npr.randn(xdim, xdim)*sc
      B    = npr.randn(xdim, udim)*sc
      C    = npr.randn(ndim, xdim)*sc
      d    = npr.randn(ndim, 1)*sc
      x0   = npr.randn(xdim,1)*sc
      U    = npr.randn(udim, cohdim, tdim)*sc
      Tio  = npr.randn(udim, tiodim, tdim)*sc
      coh  = npr.randn(udim, cohdim)*sc

    
      if not Afixed    or cx2 :  A_L.append(A)
      if not Bfixed    or cx2 :  B_L.append(B)
      if not Cfixed    or cx2 :  C_L.append(C)
      if not dfixed    or cx2 :  d_L.append(d)
      if not x0fixed   or cx2 :  x0_L.append(x0)
      if not Ufixed    or cx2 :  U_L.append(U)
      if not Tiofixed  or cx2 :  Tio_L.append(Tio)
      if not cohfixed  or cx2 :  coh_L.append(coh)
                  
      cx2 = False
      
    init_params = {'x0': x0_L, 'A': A_L, 'B': B_L, 'C': C_L, 'd': d_L, 
                   'U': U_L, 'Tio': Tio_L, 'coh': coh_L}  
    
    return init_params



def gradmask(specs):
  
    # set to ones below the parameters you want to learn and the rest to zero
    ndim, tdim, mdim, cdim, cxdim  = specs['data dims']
    udim, xdim, cohdim, tiodim     = specs['latent dims']
    
    Afixed,Bfixed,Cfixed,dfixed ,_,_,_,_      = specs['contextual constraints']
    _,_,_,_, x0fixed,Ufixed,Tiofixed,cohfixed = specs['contextual constraints']
    
    A_L   = []
    B_L   = []
    C_L   = []
    d_L   = []
    x0_L  = []
    U_L   = []
    Tio_L = []
    coh_L = []
    
    cx2  = True
    
    for cx in range(cxdim):
    
      # Here, set to ones the desired parameters to be learned. For the rest,
      # the gradients will be set to zero during the optimization process
      A    = np.ones((xdim, xdim))
      B    = np.ones((xdim, udim))
      C    = np.zeros((ndim, xdim))
      d    = np.zeros((ndim, 1))
      x0   = np.ones((xdim,1))
      U    = np.ones((udim, cohdim, tdim))
      Tio  = np.ones((udim, tiodim, tdim))
      coh  = np.ones((udim, cohdim))
    
      if not Afixed    or cx2 :  A_L.append(A)
      if not Bfixed    or cx2 :  B_L.append(B)
      if not Cfixed    or cx2 :  C_L.append(C)
      if not dfixed    or cx2 :  d_L.append(d)
      if not x0fixed   or cx2 :  x0_L.append(x0)
      if not Ufixed    or cx2 :  U_L.append(U)
      if not Tiofixed  or cx2 :  Tio_L.append(Tio)
      if not cohfixed  or cx2 :  coh_L.append(coh)
                  
      cx2 = False
      
    masked_params = {'x0': x0_L, 'A': A_L, 'B': B_L, 'C': C_L, 'd': d_L, 
                     'U': U_L, 'Tio': Tio_L, 'coh': coh_L}  
    
    return masked_params

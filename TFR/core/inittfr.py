#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize TFR model

Created on Thu Jun 13 2019

@author: jmagraner
"""

# %%

import autograd.numpy.random as npr


def initparams_random(specs,*unused):
  
    sc=0.01
    
    ndim, tdim, mdim, cdim, cxdim  = specs['data dims']
    udim, xdim, cohdim, tiodim     = specs['latent dims']
    
    ABfixed,Cfixed,Ufixed,Tiofixed,_,_,_  = specs['contextual constraints']
    _,_,_,_,cohfixed,biasfixed,biastfixed = specs['contextual constraints']
    
    AB_L  = []
    C_L   = []
    U_L   = []
    Tio_L = []
    coh_L = []
    b_L   = []
    bt_L  = []
    
    cx2  = True
    
    for cx in range(cxdim):
    
      AB   = npr.randn(xdim, tdim, udim+1)*sc
      C    = npr.randn(ndim, xdim)*sc
      U    = npr.randn(udim, cohdim, tdim)*sc
      Tio  = npr.randn(udim, tiodim, tdim)*sc
      coh  = npr.randn(udim, cohdim)*sc
      bt   = npr.randn(1,tdim)*sc
      b    = npr.randn(1)

    
      if not ABfixed    or cx2 :  AB_L.append(AB)
      if not Cfixed     or cx2 :  C_L.append(C)
      if not Ufixed     or cx2 :  U_L.append(U)
      if not Tiofixed   or cx2 :  Tio_L.append(Tio)
      if not cohfixed   or cx2 :  coh_L.append(coh)
      if not biasfixed  or cx2 :  b_L.append(b)
      if not biastfixed or cx2 :  bt_L.append(bt)
                  
      cx2 = False
      
    init_params = {'AB': AB_L, 'C': C_L, 'U': U_L, 'Tio': Tio_L,
                   'coh': coh_L, 'bias': b_L, 'biast': bt_L}  
    
    return init_params


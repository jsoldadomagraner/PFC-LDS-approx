#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:11:14 2019

@author: joana
"""

from pathlib import Path


def mypaths():
    
    homepath  = str(Path.home())

    modelpath = homepath + '/projects/Mante/fitted_models/'
    datapath  = homepath + '/projects/Mante/data/'
    
    return modelpath,datapath
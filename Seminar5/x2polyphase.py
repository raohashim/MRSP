# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 18:02:34 2016
@author: Max
modified, Gerald Schuller, July 2016
"""

import numpy as np
def x2polyphase(x,N):
    """Converts input signal x (a row vector) into a polyphase row vector 
    for blocks of length N"""      

    #Number of blocks in the signal:
    L = int(np.floor(max(np.shape(x))/N))  
    print("L= ", L)
  
    xp = np.zeros((1,N,L))
    for m in range(0,L):
        xp[0,:,m] = x[m*N+np.arange(N)]
    return xp

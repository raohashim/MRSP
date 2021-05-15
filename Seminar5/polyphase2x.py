# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:05:15 2016
@author: Max
From Matlab
Modified, Gerald Schuller, July 2016
"""
import numpy as np
def polyphase2x(xp):
	"""Converts polyphase input signal xp (a row vector) into a contiguos row vector
	For block length N, for 3D polyphase representation (exponents of z in the third 
	matrix/tensor dimension)"""
	#Number of blocks in the signal
	[r,N,L] = np.shape(xp);
	x = np.zeros((1,N*L));
	for m in range(L):
	    x[0,m*N+np.arange(N)]=xp[0,:,m]
	return x


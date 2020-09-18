import numpy as np
from activation_functions import *

#Function to calculate linear part of forward propagation
def linear_forward(A, W, b):
    
    Z=np.dot(W,A)+b
    
    assert(Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)
    
    return Z,cache

#Function to calculate activation for linear part of forward propagation
def activation_linear_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, linear_cache=linear_forward(A_prev, W, b)
        A, activation_cache=sigmoid(Z)
    
    elif activation == "tanh":
        Z, linear_cache=linear_forward(A_prev, W, b)
        A, activation_cache=tanh(Z)
        
    elif activation == "relu":
        Z, linear_cache=linear_forward(A_prev, W, b)
        A, activation_cache=relu(Z)
     
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
        
    cache=(linear_cache,activation_cache)
    
    return A,cache

#Putting it all together to form a model
def model_linear_forward(X, parameters, act):
    
    caches=[]
    A=X
    L=len(parameters)//2
    
    for l in range(1,L):
        A_prev=A
        A,cache = activation_linear_forward(A_prev,parameters['W'+str(l)], parameters['b'+str(l)],activation=act)
        caches.append(cache)
        
    AL, cache = activation_linear_forward(A, parameters['W'+str(L)], parameters['b'+str(L)],activation="sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    
    return AL, caches

        
 








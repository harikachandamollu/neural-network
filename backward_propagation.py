import numpy as np
from activation_functions import *

#Function to calculate linear part of backward propagation
def cross_entropy_linear_backward(dZ,cache):
    
    A_prev,W,b=cache
    m=A_prev.shape[1]
    
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

#Function to calculate activation for linear part of backward propagation
def cross_entropy_activation_linear_backward(dA,cache,activation):
    
    linear_cache,activation_cache=cache
    
    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev, dW, db=cross_entropy_linear_backward(dZ,linear_cache)
        
    elif activation=="tanh":
        dZ=tanh_backward(dA,activation_cache)
        dA_prev, dW, db=cross_entropy_linear_backward(dZ,linear_cache)
        
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db=cross_entropy_linear_backward(dZ,linear_cache)
        
    return dA_prev, dW, db   

#Putting it all together to form a model
def cross_entropy_model_linear_backward(AL,Y,caches,act):
    
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = cross_entropy_activation_linear_backward(dAL,current_cache,"sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = cross_entropy_activation_linear_backward(grads["dA"+str(l+1)],current_cache,act)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
        
        
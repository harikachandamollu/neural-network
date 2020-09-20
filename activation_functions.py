import numpy as np

#Function to calculate sigmoid function of a value
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache


#Function to calculate relu function of a value
def relu(Z):
    A = np.maximum(0,Z)
    cache=Z
    return A,cache


#Function to calculate tanh function of a value
def tanh(Z):
    A = np.tanh(Z)
    cache=Z
    return A,cache


#Function to calculate sigmoid function in backward propagation
def sigmoid_backward(dA,cache):
    Z=cache
    s=1/(1+np.exp(-Z))
    dZ=dA*s*(1-s)
    return dZ


#Function to calculate relu function in backward propagation
def relu_backward(dA,cache):
    Z=cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0]=0
    return dZ
    
#Function to calculate tanh function in backward propagation
def tanh_backward(dA,cache):
    Z=cache
    s=np.tanh(Z)
    dZ=dA*(1-(s*s))
    return dZ
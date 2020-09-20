import numpy as np

#Function to calculate cross entropy cost
def cross_entropy_cost(AL, Y):
    
    m = Y.shape[1]
    
    cost = -(np.dot(Y,np.log(AL.T))+np.dot(1-Y,np.log(1-AL).T))/m
    
    cost = np.squeeze(cost)
    assert(cost.shape==())
    
    return cost
import numpy as np
from forward_propagation import *
#Function to update parameters using gradient descent
def update_parameters(parameters, grads, learning_rate):
    
    L=len(parameters)//2
    
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
        
    return parameters

#Function to predict using parameters
def predict(X, y, parameters, act): 
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = model_linear_forward(X, parameters, act)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    accuracy=np.sum((p == y)/m)
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(accuracy))
        
    return p, accuracy

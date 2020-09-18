import numpy as np


#Function to initialize parameters
def initialize_parameters(layer_dims):
    
    
    #initialization
    parameters = {} #dictionary
    L = len(layer_dims)
    
    
    #Printing out which model is used
    if L==2:
        print("It uses logistic regression")
    elif L>2:
        print("It is a",L-1,"layer neural network with",L-2,"hidden layers")
    else:
        print("Please use valid list to initialize the parameters. You must be entering more than 1 value in the list where first value indicates input layer, last value indicates output layer and remaining values indicate hidden layers")
        
     
    #initializing W(weights) and b(bias) parameters with random values and zeros respectively
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters["b"+str(l)]=np.zeros((layer_dims[l],1))
        
        assert(parameters["W"+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
        assert(parameters["b" + str(l)].shape==(layer_dims[l],1))
    
    
    #returning the parameters dictionary
    return parameters
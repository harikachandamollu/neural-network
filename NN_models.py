from activation_functions import *
from initialization import *
from forward_propagation import *
from backward_propagation import *
from cost_functions import *
from updation_and_prediction import *
#L_layer_model
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, act="relu"):
    
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters(layers_dims)
        
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = model_linear_forward(X, parameters,act)
               
        # Compute cost.       
        cost = cross_entropy_cost(AL,Y)
           
        # Backward propagation.       
        grads = cross_entropy_model_linear_backward(AL, Y, caches,act)
        
        # Update parameters.       
        parameters = update_parameters(parameters, grads, learning_rate)
                       
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    #plt.plot(np.squeeze(costs))
    #plt.ylabel('cost')
    #plt.xlabel('iterations (per hundreds)')
    #plt.title("Learning rate =" + str(learning_rate))
    #plt.show()
    
    return parameters
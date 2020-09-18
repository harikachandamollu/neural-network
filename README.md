# Neural Network for classification

Structure of neural network is of the form input layer, hidden layers(0 to L), output layer

Two classification models created 
- Logistic Regression (zero hidden layers)
- Neural network (1 to L hidden layers)

## General methodology:
1. Define the neural network structure(#input_units, #hidden_units, #output_units)
2. Initialize the model's parameters
3. Loop:
    1. Implement forward propagation
    2. Compute loss
    3. Implement backward propagation to get the gradients
    4. Update parameters (gradient descent)
4. Merge 1,2,3 steps into one function

## Contents:
- __initialization.py__ : Module to initialize parameters (W,b) depending on the dimensions of the layers. layer_dimensions are passed to the function from IPython notebook. __(2. Initialize the model's parameters)__

    Modules:

        def initialize_parameters(layer_dims):

- __activation_functions.py__ : Modules to calculate activation functions like sigmoid(logistic), relu(rectified linear unit), tanh for forward and backward propagation.

    Modules:
    
        def sigmoid(Z):
        def relu(Z):
        def tanh(Z):
        def sigmoid_backward(dA,cache):
        def relu_backward(dA,cache):
        def tanh_backward(dA,cache):

- __forward_propagation.py__ : Modules to calculate forward propagation for all the layers. linear_forward carries out the calculation of multiplying input parameters and weights adding it with bias. activation_linear_forward applies acctivation function to the output of the previous function. model_linear_forward combines the above two modules by applying tanh/relu activation from 1 to L-1 layers on user selection and sigmoid function to final layer since it is a binary classification model.

    Modules:
    
         def linear_forward(A, W, b):
         def activation_linear_forward(A_prev, W, b, activation):
         def model_linear_forward(X, parameters, act):
         
- __cost_functions.py__ : Module to compute the cross entropy cost from actual and predicted label.

    Modules:
        
        def cross_entropy_cost(AL,Y):
        
- __backward_propagation.py__ : Modules to calculate gradients derived from the loss. Since the derivatives depend upon the cost function, following modules compute  dW and db to adjust the parameters with respect to learning rate.

    Modules:
    
        def cross_entropy_linear_backward(dZ,cache):
        def cross_entropy_activation_linear_backward(dA,cache,activation):
        def cross_entropy_model_linear_backward(AL,Y,caches,act):
        
- __updation_and_prediction.py__ : Parameters are updated accroding to the update rule in update_parameters module. Predict module is used to predict the test set(unseen input parameters) with the help of learnt parameters.

    Modules:
    
        def update_parameters(parameters, grads, learning_rate):
        def predict(X, y, parameters):

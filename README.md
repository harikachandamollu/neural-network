# Neural Network for classification

Structure of neural network is of the form input layer, hidden layers(0 to L), output layer

Two classification models created 
- Logistic Regression (zero hidden layers)
- Neural network (1 to L hidden layers)

## General methodology:

1. Define the neural network structure(#input_units, #hidden_units, #output_units)
2. Initialize the model's parameters
3. Loop:
    A. Implement forward propagation
    B. Compute loss
    C. Implement backward propagation to get the gradients
    D. Update parameters (gradient descent)
4. Merge 1,2,3 steps into one function


## Contents:

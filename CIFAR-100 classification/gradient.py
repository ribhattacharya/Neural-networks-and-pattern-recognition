import numpy as np
from neuralnet import Neuralnetwork

def check_grad(model, x_train, y_train, example):

    """
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train
            example: current example parameters

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    eps = 1e-2
    layer, i, j, _ = example
    w_orig = model.layers[layer].w[i,j]

    # E(w_orig + eps)
    model.layers[layer].w[i,j] = w_orig + eps
    model(x_train, y_train)
    loss1 = model.loss(model.y, y_train)

    # E(w_orig - eps)
    model.layers[layer].w[i,j] = w_orig - eps
    model(x_train, y_train)
    loss2 = model.loss(model.y, y_train)

    numerical_grad = -(loss1 - loss2)/(2*eps) # since our derivative definition is negative of the difference formula
    
    # Back propogation gradient
    model.layers[layer].w[i,j] = w_orig
    model(x_train, y_train)
    model.backward(False)
    backprop_grad = model.layers[layer].dw[i,j]
    

    print("Numerical gradient: %0.9f, Backprop gradient: %0.9f" %(numerical_grad, backprop_grad))
    print("Gradient difference: %0.9f" %abs(numerical_grad - backprop_grad))
    print("Error <= O(n2)?", abs(numerical_grad - backprop_grad) <=eps**2)



def checkGradient(x_train,y_train,config):

    subsetSize = 5  #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]
    
    # Sample examples, can change this if you want to.
    # (layer id, weight i, weight j, description)
    examples = [(1, 0, 2, "Output bias"), 
                (0, 0, 2, "Hidden bias"), 
                (1, 2, 3, "Hidden to output"), (1, 4, 5, "Hidden to output"),
                (0, 2, 3, "Input to hidden"), (0, 4, 5, "Input to hidden")]
    
    for example in examples:
        model = Neuralnetwork(config)
        print("\nGradient type: " + example[3])
        check_grad(model, x_train_sample, y_train_sample, example)
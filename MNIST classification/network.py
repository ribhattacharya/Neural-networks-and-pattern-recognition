import numpy as np
from data import *
import time
import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    return 1/(np.exp(-a) + 1)


def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    N = a.shape[0]
    return np.exp(a) / (np.sum(np.exp(a), axis=1).reshape((N,1)))


def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    tol = 1e-6
    return -np.sum(t * np.log(y+tol) + (1-t) * np.log(1-y + tol))


def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    tol = 1e-6
    return -np.sum(t * np.log(y+tol))

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss
        self.out_dim = out_dim
        self.weights = np.zeros((hyperparameters.p+1, out_dim))

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        return self.activation(X @ self.weights)

    def __call__(self, X):
        return self.forward(X)

    def train(self, minibatch):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        X, y_target = minibatch
        
        if self.out_dim == 1:
            y_target = y_target.reshape((X.shape[0],1))
        elif self.out_dim == 10:
            y_target = onehot_encode(y_target)
        
        y_pred = self.forward(X)    #(minibatch, 10)

        self.weights += self.hyperparameters.learning_rate * (X.T @ (y_target - y_pred))

    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        X, y_target = minibatch
        

        y_pred = self.forward(X)
        
        if self.out_dim == 1:       # For logistic regression
            y_target = y_target.reshape((X.shape[0],1))
            loss = self.loss(y_pred, y_target)/X.shape[0]
            y_pred = (y_pred > 0.5).astype(int)     # y_pred as binary (0/1)
            acc = np.mean(y_pred.reshape((X.shape[0],1)) == y_target)
        
        elif self.out_dim == 10:    # For softmax regression
            loss = self.loss(y_pred, onehot_encode(y_target))/X.shape[0]
            acc = np.mean(onehot_decode(y_pred) == y_target)

        return loss, acc
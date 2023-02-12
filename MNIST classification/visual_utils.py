from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def plotPCAComp(components):

    '''
    A function to plot the PCA components in the form of images.

    Input
    -----
    The components to be plotted

    Output
    ------
    Gray-scale images
    '''

    fig, ax = plt.subplots(5,2, figsize=(10,10))
    fig.suptitle('Top 10 PCA components')
    
    for i in range(10):
        r,c = np.unravel_index(i, (5,2))
        ax[r,c].imshow(components[i].reshape((28,28)), cmap='gray')
        ax[r,c].set_title('# %d' %i)
        ax[r,c].set_axis_off()
    
    fig.tight_layout()
    plt.show()


def plotLOSS(LOSS_train, LOSS_valid, hyperparameters):

    '''
    A function to plot the loss curves for training and validation set

    Innput
    ------
    LOSS_train: mean training loss over k-folds
    LOSS_valid: mean validation loss over k-folds
    hyperparameters: training parameters

    Output
    -------
    Training loss and Validation loss curves with respect to epochs
    '''
    
    plt.figure()
    if hyperparameters.regression_type == 'LR':
        plt.title('Logistic regression loss for %d vs %d' %(hyperparameters.digit1,hyperparameters.digit2))
    elif hyperparameters.regression_type == 'SR':
        plt.title('Softmax regression loss')
    
    plt.plot(LOSS_train, color='red', label='Training loss')
    plt.plot(LOSS_valid, color='blue', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Average loss over k-folds')
    plt.legend()
    plt.show()


def plotWeights(w, components):

    '''
    A function to plot the trained weights as gray-scale images

    Input
    ------
    w: The trained weights
    components: PCA components to reproject the weights

    Output
    -------
    Trained weights in the form of gray-scale images to visualize features.
    '''

    w = w[1:,:]           # (p, 10)
    w_proj = w.T @ components    # (10, 784)

    fig, ax = plt.subplots(5,2, figsize=(10,10))
    fig.suptitle('Reprojected weights')
    
    for i in range(10):
        r,c = np.unravel_index(i, (5,2))
        ax[r,c].imshow(w_proj[i].reshape((28,28)), cmap='gray')
        ax[r,c].set_title('# %d' %i)
        ax[r,c].set_axis_off()
    
    fig.tight_layout()

    plt.show()


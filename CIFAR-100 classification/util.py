import copy
import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants
from tqdm import tqdm


def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)



def normalize_data(inp):
    """
    Normalizes inputs (on per channel basis of every image) here to have 0 mean and unit variance.
    This will require reshaping to seprate the channels and then undoing it while returning

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions

    returns:
        normalized inp: N X d 2D array

    """
    inp_r = inp[:,:32*32]
    inp_g = inp[:,32*32:2*32*32]
    inp_b = inp[:,2*32*32:]

    mu_r = np.mean(inp_r, axis=1).reshape(-1,1)
    mu_g = np.mean(inp_g, axis=1).reshape(-1,1)
    mu_b = np.mean(inp_b, axis=1).reshape(-1,1)

    sd_r = np.std(inp_r, axis=1).reshape(-1,1)
    sd_g = np.std(inp_g, axis=1).reshape(-1,1)
    sd_b = np.std(inp_b, axis=1).reshape(-1,1)
    tol = 0

    return np.hstack([(inp_r - mu_r)/(sd_r + tol), (inp_g - mu_g)/(sd_g + tol), (inp_b - mu_b)/(sd_b + tol)])


def one_hot_encoding(labels, num_classes=20):
    """
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (20/100 for CIFAR-100)

    returns:
        oneHot : N X num_classes 2D array

    """
    n = labels.shape[0]
    encode = np.zeros((n,num_classes))
    encode[np.arange(n), labels.astype(int)] = 1

    return encode


def generate_minibatches(dataset, batch_size=64):
    """
        Generates minibatches of the dataset

        args:
            dataset : 2D Array N (examples) X d (dimensions)
            batch_size: mini batch size. Default value=64

        yields:
            (X,y) tuple of size=batch_size

        """

    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def calculateCorrect(y,t):  #Feel free to use this function to return accuracy instead of number of correct predictions
    """
    Calculates the number of correct predictions

    args:
        y: Predicted Probabilities
        t: Labels in one hot encoding

    returns:
        accuracy percentage
    """
    t = np.argmax(t, axis=1)
    y = np.argmax(y, axis=1)

    return np.mean(y == t) * 100



def append_bias(X):
    """
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])

def shuffle(dataset):
    """
    Shuffle dataset.

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    
    X,y = dataset
    n = X.shape[0]

    data = np.hstack([X, y.reshape((n,1))])

    np.random.shuffle(data)

    X = data[...,:-1]
    y = data[...,-1]

    return (X,y)


def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):

    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. earlyStop=-1 means the last epoch was the best one
    """

    fig1, ax1 = plt.subplots(figsize=((16, 10)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10))
    plt.yticks()
    ax1.set_title('Loss Plots')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.legend(loc="upper right")
    plt.savefig(constants.saveLocation+"loss.png")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((16, 10)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10))
    plt.yticks()
    ax2.set_title('Accuracy Plots')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc="lower right")
    plt.savefig(constants.saveLocation+"accuarcy.png")
    plt.show()

    #Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.saveLocation+"trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.saveLocation+"valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(constants.saveLocation+"trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(constants.saveLocation+"valEpochAccuracy.csv")



def createTrainValSplit(x_train,y_train):

    """
    Creates the train-validation split (80-20 split for train-val). Please shuffle the data before creating the train-val split.
    """
    x_train, y_train = shuffle((x_train, y_train))
    n = int(0.2 * x_train.shape[0])
    
    return x_train[:-n], y_train[:-n], x_train[-n:], y_train[-n:]



def load_data(path, num_classes):
    """
    Loads, splits our dataset- CIFAR-100 into train, val and test sets and normalizes them

    args:
        path: Path to cifar-100 dataset
        num_classes: No. of classes for classification
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """

    if num_classes == 20:   # change dataset for 20 vs 100 classification
        dict_name = b'coarse_labels'
    elif num_classes == 100:
        dict_name = b'fine_labels'

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, constants.cifar100_directory)

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    images_dict = unpickle(os.path.join(cifar_path, "train"))
    data = images_dict[b'data']
    label = images_dict[dict_name]
    train_labels.extend(label)
    train_images.extend(data)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels).reshape((len(train_labels),-1))
    train_images, train_labels, val_images, val_labels = createTrainValSplit(train_images,train_labels)
    

    train_normalized_images = normalize_data(train_images)
    train_one_hot_labels = one_hot_encoding(train_labels, num_classes)

    val_normalized_images = normalize_data(val_images)
    val_one_hot_labels = one_hot_encoding(val_labels, num_classes)


    test_images_dict = unpickle(os.path.join(cifar_path, "test"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[dict_name]
    test_images = np.array(test_data)
    test_labels = np.array(test_labels)

    test_normalized_images= normalize_data(test_images)
    test_one_hot_labels = one_hot_encoding(test_labels, num_classes)


    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

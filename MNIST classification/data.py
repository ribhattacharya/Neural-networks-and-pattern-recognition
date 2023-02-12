import idx2numpy
import numpy as np
import os
import pickle


def getTwoLabels(X, y, n1, n2):
    """
    Return subset of data for two labels n1 and n2.

    Parameters
    ----------

    X: numpy.ndarray
        Feature vector for labels n1 and n2.
    
    y: numpy.array
        labels for n1 and n2 (binary 0/1).

    n1: int
        label 1
    
    n2: int
        label 2

        Returns
    -------
        Tuple: (numpy.ndarray, numpy.array)
            Subset (X,y)
    """

    X = X[np.where(np.logical_or(y == n1 , y == n2))]
    y = y[np.where(np.logical_or(y == n1 , y == n2))]
    y = (y == n2).astype(int)
    
    return X,y

def dataPreProcess(X, pca):
    """
    Return processed data for either regression.

    Parameters
    ----------

    X: numpy.ndarray
        The image data (28*28).
    

    pca: pca object
        pca object used for dimensionality reduction.

        Returns
    -------
        X: numpy.ndarray
            Transformed dataset X
    """

    X = pca.transform(X)
    X = append_bias(X)
    
    return X

def load_data(data_directory, train = True):
    if train:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'train_images'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'train_labels'))
    else:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'test_images'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'test_labels'))

    vdim = images.shape[1] * images.shape[2]
    vectors = np.empty([images.shape[0], vdim])
    for imnum in range(images.shape[0]):
        imvec = images[imnum, :, :].reshape(vdim, 1).squeeze()
        vectors[imnum, :] = imvec
    
    return vectors, labels

def z_score_normalize(X, u = None, sd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)

    return ((X - mu)/(sd + 1e-6), mu, sd)


def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    _min = (np.min(X, axis = 0)).reshape((1,X.shape[1]))
    _max = (np.max(X, axis=0)).reshape((1,X.shape[1]))
    return ((X - _min)/(_max - _min + 1e-6), _min, _max)


def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    n = y.shape[0]
    k = 10
    _encode = np.zeros((n,k))
    _encode[np.arange(n), y.astype(int)] = 1

    return _encode


def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    return np.argmax(y, axis=1)


def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

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


def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape ((N+1),d)
    """

    return np.hstack([np.ones(X.shape[0]).reshape((X.shape[0], 1)), X])


def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]

def generate_k_fold_set(dataset, k = 5): 
    X, y = dataset
    if k == 1:
        yield (X, y), (X[len(X):], y[len(y):])
        return

    order = np.random.permutation(len(X))
    
    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width
    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width



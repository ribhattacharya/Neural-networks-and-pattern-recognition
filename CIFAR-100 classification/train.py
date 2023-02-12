
import copy
from neuralnet import *

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """

    # Read in the esssential configs

    patience = config["early_stop_epoch"]
    
    train_loss_arr = np.zeros((config["epochs"], 1))
    train_acc_arr = np.zeros((config["epochs"], 1))
    valid_loss_arr = np.zeros((config["epochs"], 1))
    valid_acc_arr = np.zeros((config["epochs"], 1))

    early_arr = np.zeros(patience)
    best_model = {}
    earlyStop = -1
    
    for i in tqdm(range(config["epochs"])):
        
        for (X,y) in generate_minibatches((x_train, y_train), config["batch_size"]):
            model(X,y)
            model.backward()

        best_model[i] = copy.deepcopy(model)

        train_acc_arr[i], train_loss_arr[i] = modelTest(model, x_train, y_train)
        valid_acc_arr[i], valid_loss_arr[i] = modelTest(model, x_valid, y_valid)

        print("Train acc: %f and loss: %f"%(train_acc_arr[i], train_loss_arr[i]))
        print("Valid acc: %f and loss: %f"%(valid_acc_arr[i], valid_loss_arr[i]))

        print("early stop arr: ", early_arr)
        
        if i < patience:        # Store first 5 validation losses into array
            early_arr[i] = valid_loss_arr[i]
        
        elif earlyStop == -1:   # if earlyStop has not been found
            temp = valid_loss_arr[i]
        
            if (temp > early_arr).all():    # If loss > previous 5 losses, then earlystop
                earlyStop = i-np.argmin(early_arr)-1
                print("Early stop at %d,  Valid acc: %f%% and loss: %f"%(earlyStop, valid_acc_arr[earlyStop], valid_loss_arr[earlyStop]))
                # break

            else:                           # else push new iteration loss into early stop array
                early_arr = np.hstack([early_arr[1:], temp])
        

    plots(train_loss_arr, train_acc_arr, valid_loss_arr, valid_acc_arr, earlyStop)
        
    if earlyStop == -1:
        earlyStop = config["epochs"] - 1 
    
    return best_model[earlyStop]

#This is the test method
def modelTest(model, X_test, y_test):
    """
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    loss, acc = model(X_test, y_test)
    return acc, loss



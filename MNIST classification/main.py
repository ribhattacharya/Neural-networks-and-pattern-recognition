import argparse
from data import *
from network import *
from image import *
from tqdm import tqdm
from visual_utils import *




def main(hyperparameters):

	'''
	Main function to run the classification algorithms

	Input
	------
	Hyperparameters: The hyperparameters passed as arguments while running the main.py file

	Output
	-------
	Loss curves, component images, weight images, testing accuracy
	'''

	# Load data, normalize and apply PCA
	data_train, labels_train = load_data('./data', train=True)		# 60k examples
	data_test, labels_test = load_data('./data', train=False)		# 10k examples
	data_train, *_ = hyperparameters.normalization(data_train)
	data_test, *_ = hyperparameters.normalization(data_test)
	
	# Preprocess data and set parameters according to LR or SR
	if hyperparameters.regression_type == 'LR':
		digit_1, digit_2 = hyperparameters.digit1, hyperparameters.digit2 
		data_train, labels_train = getTwoLabels(data_train, labels_train, digit_1, digit_2)
		data_test, labels_test = getTwoLabels(data_test, labels_test, digit_1, digit_2)
		activation, loss, out_dims = sigmoid, binary_cross_entropy, 1
	
	elif hyperparameters.regression_type == 'SR':
		activation, loss, out_dims = softmax, multiclass_cross_entropy, 10

	# Implement PCA
	pca = PCA(n_components=hyperparameters.p)
	pca.fit(data_train)

	# Initialize loss and weight arrays to store model data
	LOSS_train = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
	LOSS_valid = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
	weights_arr = np.zeros((hyperparameters.epochs, hyperparameters.p+1, out_dims))

	# Loop over k-folds -> epocs -> minibatches
	for k, k_fold_data in enumerate(generate_k_fold_set((data_train, labels_train), hyperparameters.k_folds)):
		print("Training fold # %d of %d" %(k+1, hyperparameters.k_folds))

		# Get fold data from provided functions and PCA transform
		((X_train, y_train), (X_valid, y_valid)) = k_fold_data
		X_train = dataPreProcess(X_train, pca)
		X_valid = dataPreProcess(X_valid, pca)

		# Initialize neural net for k-th fold
		net =  Network(hyperparameters, activation, loss, out_dims)

		for t in (range(hyperparameters.epochs)):
			X_train, y_train = shuffle((X_train, y_train))	# Shuffle data before training

			# Train data in minibatch
			for train_minibatch in generate_minibatches((X_train, y_train), hyperparameters.batch_size):
				net.train(train_minibatch)
			
			# Store loss data and weights for plotting and early stopping
			LOSS_train[k,t] = net.test((X_train, y_train))[0]
			LOSS_valid[k,t] = net.test((X_valid, y_valid))[0]
			weights_arr[t] += net.weights		# Add weights now within loop, average during post processing
	
	# Mean of losses and weights over all folds
	LOSS_train = np.mean(LOSS_train, axis=0)
	LOSS_valid = np.mean(LOSS_valid, axis=0)
	weights_arr /= hyperparameters.k_folds # divide by k_folds since we added earlier inside loop
	
	# Compute epoch with minimum validation loss (over k-folds) and retrive its weights
	min_loss_epoc = np.argmin(LOSS_valid)
	net.weights = weights_arr[min_loss_epoc]
	print("\nMinimum loss on validation data found in epoch # %d" %min_loss_epoc)

	# Compute performance of model over unseen test data
	data_test = dataPreProcess(data_test, pca)
	loss_test, acc_test = net.test((data_test, labels_test))
	print("\nTesting loss: %.4f, Testing accuracy: %.2f %%\n" %(loss_test, acc_test*100))

	# Plots and visualizations
	plotLOSS(LOSS_train, LOSS_valid, hyperparameters)
	plotPCAComp(pca.components_)

	# Plot weights if softmax regression
	if hyperparameters.regression_type == 'SR':
		plotWeights(net.weights, pca.components_)
		

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'CSE251B PA1')
    parser.add_argument('--batch-size', type = int, default = 1,
	    help = 'input batch size for training (default: 1)')
    parser.add_argument('--epochs', type = int, default = 100,
	    help = 'number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type = float, default = 0.001,
	    help = 'learning rate (default: 0.001)')
    parser.add_argument('--k-folds', type = int, default = 5,
	    help = 'number of folds for cross-validation')
    parser.add_argument('--p', type = int, default = 200,
	    help = 'number of principal components')
    parser.add_argument('--regression-type', type = str, default = 'LR',
		help = 'Regression type - Logistic Regression (LR) or Softmax Regression (SR)')
    parser.add_argument('--digit1', type = int, default = 2,
		help = 'Digit 1 to use (only for LR)')
    parser.add_argument('--digit2', type = int, default = 7,
		help = 'Digit 2 to use (only for LR)')
    parser.add_argument('--z-score', dest = 'normalization', action='store_const', 
	    default = min_max_normalize, const = z_score_normalize,
	    help = 'use z-score normalization on the dataset, default is min-max normalization')

    hyperparameters = parser.parse_args()

    main(hyperparameters)

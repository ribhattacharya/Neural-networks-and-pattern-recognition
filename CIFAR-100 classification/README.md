# Multiclass classification on CIFAR-100 dataset using back-propagation

The program can be run directly from the CLI.

To run the classification models (with our tuned hyperparameters), type following commands in your terminal (opened in same directory as the code files):

## Testing gradients (config_3b.yaml)
     python3 main.py --experiment test_gradients

Change config_3b.yaml to change parameters for the experiment.

## Testing gradients (config_3c.yaml)
     python3 main.py --experiment test_momentum

Change config_3c.yaml to change parameters for the experiment.

## Testing gradients (config_3d.yaml)
     python3 main.py --experiment test_regularization

Change config_3d.yaml to change parameters for the experiment.

## Testing gradients (config_3e.yaml)
     python3 main.py --experiment test_activation

Change config_3e.yaml to change parameters for the experiment.

## Testing gradients (config_3f_i.yaml)
     python3 main.py --experiment test_hidden_units

Change config_3f_i.yaml to change parameters for the experiment.

## Testing gradients (config_3f_ii.yaml)
     python3 main.py --experiment test_hidden_layers

Change config_3f_ii.yaml to change parameters for the experiment.

## Testing gradients (config_3g.yaml)
     python3 main.py --experiment test_100_classes

Change config_3g.yaml to change parameters for the experiment.
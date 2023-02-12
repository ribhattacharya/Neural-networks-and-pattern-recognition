The program can be run directly from the CLI. The arguments required by the proram (optional) are 

     usage: main.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--learning-rate LEARNING_RATE] [--k-folds K_FOLDS] [--p P][--regression-type REGRESSION_TYPE] [--digit1 DIGIT1] [--digit2 DIGIT2] [--z-score]

To run the classification models (with our tuned hyperparameters), type following commands in your terminal (opened in same directory as the code files):

# Logistic Regression
## 2v7 model
     python3 main.py --batch-size 50 --epochs 100 --learning-rate 0.009 --k-folds 10 --p 200 --regression-type LR --digit1 2 --digit2 7
   
## 5v8 model
     python3 main.py --batch-size 25 --epochs 100 --learning-rate 0.006 --k-folds 10 --p 200 --regression-type LR --digit1 5 --digit2 8

# Softmax Regression
     python3 main.py --batch-size 50 --epochs 100 --learning-rate 0.007 --k-folds 10 --p 200 --regression-type SR
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 3, suppress = True)

from model import Multiple_Linear_Regression as LR
from loss import MSEloss
from scaler import StandardScaler
from trainer import Trainer
from plot_train_result import plot_history, plot_truth_and_prediction

from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    ########################
    # Hyperparameters
    ########################
    n_sample = 100
    n_iter = 200
    lr = 5e-2

    ########################
    # Load data
    ########################
    # 1) features
    x1 = np.random.randn(n_sample,) * 2         # feature 1
    x2 = np.random.randn(n_sample,) * 5         # feature 2
    x3 = np.random.randn(n_sample,) * 3         # feature 3
    X = np.c_[x1, x1**2, x2, x2**2, x3, x3**2]  # features for multiple linear regression
    n_feature = X.shape[-1]

    # 2) true weights
    w_true = np.array([0.5, 0.6, 0.2, -1.2, 1.3, -0.9])
    b_true = 2.1

    # 3) targets
    y = np.dot(X, w_true) + b_true
    y += np.random.randn(*y.shape) * 0.01

    # 4) train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    ########################
    # Standardize features
    ########################
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)      # standardized train features
    X_test_norm = (X_test - scaler.mean) / scaler.sd  # standardized test features
    #X_test_norm = scaler.fit_transform(X_test)       # standardized test features

    ########################
    # Define model, loss & trainer
    ########################
    model = LR(n_weights = n_feature)
    loss_fn = MSEloss()
    trainer = Trainer(model, loss_fn, n_iter, lr)

    ########################
    # Train model
    ########################
    history = trainer.train(X_train_norm, y_train, X_test_norm, y_test)

    ########################
    # Print final result
    ########################
    # Final weights
    print("\n----- Trained weights -----")
    for k, v in model.params.items():
        print(k, ":", v)
    print()

    # Final test loss
    y_test_pred = model(X_test_norm)                 # prediction for test data
    test_loss = loss_fn(y_test_pred, y_test)         # MSE loss for test data
    print("\n----- Final test loss -----")
    print(f"{test_loss:.2f}")

    ########################
    # Visualization
    ########################
    # 1) Train/test loss history
    plot_history(history, save = True, fname = "./figures/Train_History.png")

    # 2) Test data prediction
    plot_truth_and_prediction(
        X_test, y_test, y_test_pred, title = "Test Data - Truth vs. Prediction", 
        save = True, fname = "./figures/Test_Data_Truth_vs_Prediction.png"
    )
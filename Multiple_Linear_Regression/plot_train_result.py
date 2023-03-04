import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, save = True, fname = "Train_History.png"):
    '''
    Plot train/test loss history

    Parameters
    -----
    history : train record dictionary
        - keys = 
          'train_loss' : train loss at each iteration
                        - shape = (n_iter,) | list
          'test_loss' : test loss at each 10-th iteration
                        - shape = (n_iter//10,) | list
    save : whether to save figure
        - bool
    fname : figure file name
        - string
    '''

    n_train_record = len(history['train_loss'])
    n_test_record = len(history['test_loss'])

    train_iters = np.arange(n_train_record)
    test_iters = np.arange(n_test_record) * n_train_record / n_test_record

    fig = plt.figure()
    plt.plot(train_iters, history['train_loss'], 'ko-', markersize = 2.5, label = 'Train')
    plt.plot(test_iters, history['test_loss'], 'ro-', markersize = 2.5, label = 'Test')
    plt.legend()
    plt.xlabel("Iteration", fontsize = 11)
    plt.ylabel("Loss", fontsize = 11)
    plt.title("Train & Test Loss", fontweight = 'bold', fontsize = 15)
    if save:
        plt.savefig(fname)
    else:
        plt.show()

def plot_truth_and_prediction(X, y, y_pred, title = "Truth vs. Prediction", 
                              save = True, fname = "Truth_vs_Prediction.png"):
    '''
    Plot each feature vs. (truth/prediction) scatter plot

    Parameters
    -----
    X : features
        - shape = (n_sample, n_feature) | np.ndarray
    y : targets
        - shape = (n_sample,) | np.ndarray
    y_pred : predictions
        - shape = (n_sample,) | np.ndarray
    save : whether to save figure
        - bool
    fname : figure file name
        - string
    '''
    nrow = 2
    ncol = X.shape[-1] // 2

    fig, axes = plt.subplots(nrow, ncol, figsize = (ncol * 4, nrow * 3.5))
    plt.subplots_adjust(
        left = 0.1, right = 0.9, wspace = 0.3,
        bottom = 0.1, top = 0.9, hspace = 0.3
    )

    for i, ax in enumerate(axes.flat):
        ax.scatter(X[:,i], y, c = 'k', marker = 'o', label = 'Target')
        ax.scatter(X[:,i], y_pred, c = 'r', marker = 'x', label = 'Pred')
        ax.legend()
        ax.set_xlabel(f"Feature {i+1}", fontsize = 8)

    plt.suptitle(title, fontweight = 'bold', fontsize = 15)
    if save:
        plt.savefig(fname)
    else:
        plt.show()

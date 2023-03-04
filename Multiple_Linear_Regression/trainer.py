import numpy as np
from loss import MSEloss

class Trainer:
    def __init__(self, model, loss_fn, n_iter = 1000, lr = 1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.n_iter = n_iter
        self.lr = lr

    def train(self, X_train, y_train, X_test, y_test):
        '''
        Train the model using given data

        Parameters
        -----
        X_train : train data features
            - shape = (n_train, n_feature) | np.ndarray
        y_train : train data target
            - shape = (n_train,) | np.ndarray
        X_test : test data features
            - shape = (n_test, n_feature) | np.ndarray
        y_test : test data target
            - shape = (n_test,) | np.ndarray

        Returns
        -----
        history : train record dictionary
            - keys = 
              'train_loss' : train loss at each iteration
                            - shape = (n_iter,) | list
              'test_loss' : test loss at each 10-th iteration
                            - shape = (n_iter//10,) | list
        '''
        n_train = X_train.shape[0]

        history = {}
        history['train_loss'] = []
        history['test_loss'] = []

        for iter in range(self.n_iter):

            # forward
            y_train_pred = self.model(X_train)

            # train loss
            train_loss = self.loss_fn(y_train_pred, y_train)
            history['train_loss'].append(train_loss)

            if (iter+1) % 10 == 0:
                # test loss
                y_test_pred = self.model(X_test)
                test_loss = self.loss_fn(y_test_pred, y_test)
                history['test_loss'].append(test_loss)

                # print
                print(f"Iteration {iter+1:4d} : train loss = {train_loss:10.3f} | test loss = {test_loss:10.3f}")

            # backward (... assuming linear regression model & MSEloss)
            grads = {}
            grads['w'] = np.dot(y_train_pred - y_train, X_train) / n_train   # shape = (n_feature,)
            grads['b'] = np.sum(y_train_pred - y_train) / n_train      # shape = (,)
            
            # update
            self.model.params['w'] -= self.lr * grads['w']
            self.model.params['b'] -= self.lr * grads['b']

        return history
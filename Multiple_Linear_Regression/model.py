import numpy as np

class Multiple_Linear_Regression:
    def __init__(self, n_weights):
        self.params = {}
        self.params['w'] = np.random.randn(n_weights)
        self.params['b'] = 0
    
    def __call__(self, X):
        '''
        Predict targets using current weights

        Parameters
        -----
        X : train data features
            - shape = (n_sample, n_feature) | np.ndarray

        Returns
        -----
        y_pred : predicted targets
            - shape = (n_sample,) | np.ndarray
        '''
        y_pred = np.dot(X, self.params['w']) + self.params['b']
        return y_pred
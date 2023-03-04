import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.sd = None

    def fit_transform(self, X):
        '''
        Standardize each feature (i.e. column of X)
        i.e. x_j --> (x_j - mean(x_j)) / sd(x_j)

        Parameters
        -----
        X : features
            - shape = (n_sample, n_feature) | np.ndarray

        Returns
        -----
        X_norm : standardized features
            - shape = (n_sample, n_feature) | np.ndarray
        '''

        self.mean = X.mean(axis = 0) # mean of each feature (shape = (n_feature,))
        self.sd = X.std(axis = 0)    # sd of each feature (shape = (n_feature,))

        X_norm = (X - self.mean) / self.sd # standardized features

        return X_norm
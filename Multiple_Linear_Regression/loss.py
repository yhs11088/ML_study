import numpy as np

class MSEloss:
    def __init__(self):
        pass

    def __call__(self, y_pred, y):
        '''
        Calculate MSE loss

        Parameters
        -----
        y_pred : predicted target
            - shape = (n_sample,) | np.ndarray
        y : true target
            - shape = (n_sample,) | np.ndarray

        Returns
        -----
        loss : MSE loss
            - float
        '''
        n_sample = y.shape[0]
        loss = np.sum((y_pred - y)**2) / (2 * n_sample)
        return loss

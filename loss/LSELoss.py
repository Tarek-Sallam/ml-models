import numpy as np

class LSELoss:
    def __call__(self, y_pred, y):
        return np.sum((y_pred - y) ** 2)
    
    def grads(self, y_pred, y):
        return 2 * (y_pred - y)
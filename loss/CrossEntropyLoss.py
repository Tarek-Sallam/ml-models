import numpy as np

class BinaryCrossEntropyLoss:
    def __call__(self, y_pred, y):
        return -np.mean(y * np.log(y_pred + 1e-7) + (1 - y)*np.log(1-y_pred + 1e-7))
    
    def grads(self, y_pred, y):
        return  (-1/y.shape[0]) * (y - y_pred)
    
class CategoricalCrossEntropyLoss:
    def __call__(self, y_pred, y):
        return np.log(y_pred)[np.arange(len(y)), y]
import numpy as np

class LSELoss:
    def __call__(self, y, y_pred):
        return np.sum((y_pred - y) ** 2)
    
    def grads(self, X, y, y_pred, model_grads):
        return np.dot(model_grads.T, 2 * (y_pred - y))
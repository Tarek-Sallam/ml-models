import numpy as np

class MSELoss:
    def __call__(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)
    
    def grads(self, y_pred, y, model_grads):
        return np.dot(model_grads.T, (2 / y.shape[0]) * (y_pred - y))
import numpy as np

class MSELoss:
    def __call__(self, y, y_pred):
        return np.mean((y_pred - y) ** 2)
    
    def grads(self, X, y, y_pred, model_grads):
        return np.dot(model_grads.T, (2 / y.shape[0]) * (y_pred - y))
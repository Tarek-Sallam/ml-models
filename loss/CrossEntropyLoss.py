import numpy as np

class BinaryCrossEntropyLoss:
    def __call__(self, y, y_pred):
        return -np.sum(y * np.log(y_pred + 1e-7) + (1 - y)*np.log(1-y_pred + 1e-7))
    
    def grads(self, X, y, y_pred, model_grads):
        return np.dot(model_grads.T, (-1) * ((y - y_pred) / (y_pred - y_pred**2)))
    
class CategoricalCrossEntropyLoss:
    def __call__(self, y, y_pred):
        return -np.sum(np.log(y_pred)[np.arange(len(y)), y])
    
    def grads(self, X, y, y_pred, model_grads):
        y_pred = y_pred[np.arange(len(y)), y]
        return np.dot(model_grads, - 1 / y_pred)
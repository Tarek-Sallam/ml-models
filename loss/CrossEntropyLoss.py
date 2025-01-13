import numpy as np

class BinaryCrossEntropyLoss:
    def __call__(self, y_pred, y):
        return -np.sum(y * np.log(y_pred + 1e-7) + (1 - y)*np.log(1-y_pred + 1e-7))
    
    def grads(self, y_pred, y, model_grads):
        #print(model_grads)
        #print(np.dot(model_grads.T, (-1) * ((y - y_pred) / (y_pred - y_pred**2))))
        return np.dot(model_grads.T, (-1) * ((y - y_pred) / (y_pred - y_pred**2)))
    
class CategoricalCrossEntropyLoss:
    def __call__(self, y_pred, y):
        return np.log(y_pred)[np.arange(len(y)), y]
    
    def grads(self, y_pred, y):
        grads_matrix = np.copy(y_pred)
        for i, col in enumerate(y):
            grads_matrix[i, :] = -y_pred[i, :] 
            grads_matrix[i, col] = 1 - y_pred[i, col]
        return grads_matrix
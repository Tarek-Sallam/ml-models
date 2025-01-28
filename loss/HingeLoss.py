import numpy as np

class HingeLoss:
    def __call__(self, y, y_pred):
        return np.mean([np.max([0, 1 - y_pred_i * y_i]) for y_i, y_pred_i in zip(y, y_pred)])
        
    def grads(self, y, y_pred, loss, model_grads):
        return [np.mean([-y_i * grad_i] if (1 - y_i*y_pred_i) > 0 else 0 for y_i, y_pred_i, grad_i in zip(y, y_pred, grad)) for grad in model_grads]
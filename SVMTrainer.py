import numpy as np

class SVMTrainer:
    def __init__(self, model, optimizer, loss=None):
        self.model = model
        self.loss = loss 
        self.optimizer = optimizer
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            if not self.model.is_dual:
                y_pred = self.model(X)
                loss = self.loss(y, y_pred)
                model_grads = self.model.grads(X, y, y_pred)
                grads = self.loss.grads(X, y, y_pred, model_grads)
                params = self.model.get_params()
                self.optimizer.step(params, grads)
                self.model.set_params(params)
            else:
                grads = self.model.grads(X, y)
                params = self.model.get_params()
                self.optimizer.step(params, grads)
                self.model.set_params(params, y)
                
            print(f"Epoch: {epoch + 1}")
            print(f"Loss: {loss}")
        if self.model.is_dual:
            pass
        

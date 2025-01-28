import numpy as np

class SVMTrainer:
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss 
        self.optimizer = optimizer
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            if self.model.mode == "primal":
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
                self.model.set_params(params)
                
            print(f"Epoch: {epoch + 1}")
            print(f"Loss: {loss}")
        if self.model.mode == "dual":
            pass
        

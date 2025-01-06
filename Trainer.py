class Trainer:
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.model.forward(X)
            loss = self.loss(y_pred, y)
            loss_grads = self.loss.grads(y_pred, y)
            grads = self.model.grads(X, loss_grads)
            params = self.model.get_params()
            self.optimizer.step(params, grads)
            self.model.set_params(params)
            print(f"Epoch: {epoch + 1}")
            print(f"Loss: {loss}")
        

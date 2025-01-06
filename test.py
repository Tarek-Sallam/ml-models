from Trainer import Trainer
from optimizers.SGDOptimizer import SGDOptimizer
from models.Regression import LinearRegression
from loss.MSELoss import MSELoss
import numpy as np
import matplotlib.pyplot as plt

# GENERATE DATA
x = np.linspace(0, 10, 10)
y = 2 * x + 2 + np.sin(4 * x)

X = x.reshape((x.shape[0], 1))
model = LinearRegression(x.ndim)
optimizer = SGDOptimizer(0.01)
loss = MSELoss()
trainer = Trainer(model, loss, optimizer)

trainer.train(X, y, 10)

fig, ax = plt.subplots()

ax.plot(x, y, 'o')
y_pred = trainer.model.forward(X)
ax.plot(x, y_pred)
plt.show()
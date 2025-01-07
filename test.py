from Trainer import Trainer
from optimizers.SGDOptimizer import SGDOptimizer
from models.Regression import LinearRegression
from loss.MSELoss import MSELoss
import numpy as np
import matplotlib.pyplot as plt

# GENERATE DATA
x = np.linspace(-10, 10, 20)
y = x**3 + 1

X = x.reshape((x.shape[0],1))

model = LinearRegression(1, 3)
optimizer = SGDOptimizer(0.0000001)
loss = MSELoss()
trainer = Trainer(model, loss, optimizer)

trainer.train(X, y, 100)
fig, ax = plt.subplots()

ax.plot(x, y, 'o')
y_pred = trainer.model.forward(X)
ax.plot(x, y_pred)
plt.show()
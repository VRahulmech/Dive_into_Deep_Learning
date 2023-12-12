import torch
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,101)
y = (x-1)*(x-2)

plt.plot(x,y)
plt.show()

t = torch.tensor(2.0, requires_grad=True)
y = (t-1)*(t-2)

y.backward()
print(t.grad)


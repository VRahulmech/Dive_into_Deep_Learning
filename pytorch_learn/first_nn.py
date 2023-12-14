import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# % matplotlib inline
# import seaborn as sns


df = pd.read_csv('https://gist.githubusercontent.com/ZeccaLehn/4e06d2575eb9589dbe8c365d61cb056c/raw/64f1660f38ef523b2a1a13be77b002b98665cdfe/mtcars.csv')

print(df.columns)
print(df.info())
print(df.shape)

plt.scatter(df['wt'], df['mpg'])
plt.show()

x_list = np.array(df['wt'], dtype=np.float32).reshape(-1,1)
y_list = df['mpg'].to_list()


x_train = torch.from_numpy(x_list)
y_train = torch.tensor(y_list, requires_grad=True).reshape(-1,1)

class LinearRegressor(nn.Module):
    def __init__(self, in_, out_):
        super(LinearRegressor, self).__init__()
        self.__in = in_
        self.__out = out_
        self.linear = nn.Linear(self.__in, self.__out)
    
    def forward(self, x):
        return self.linear(x)
    


model = LinearRegressor(1,1)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.02)

max_epoch = 1000

for i in range(max_epoch):
    optimizer.zero_grad()
    y_pred = model(x_train)
    ls = loss(y_train, y_pred)
    ls.backward()
    optimizer.step()

    if i%50==0:
        with torch.no_grad():
            y_pred_test = model(x_train)
            lst = loss(y_train, y_pred_test)
            print(np.mean(np.array(lst)))


plt.scatter(df['wt'], df['mpg'])
plt.plot(df['wt'], model(x_train).detach().numpy(), color='r')
plt.show()





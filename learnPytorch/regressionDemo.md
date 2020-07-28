[toc]

# Pytorch 实现线性回归预测波士顿房价

## 导入必要包并加载数据

```
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import torch

boston = load_boston()
xInput = boston.data
yInput = boston.target
```

## 构建网络

```
class Net(torch.nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)
```

## 定义输入输出及优化器

```
x = torch.tensor(xInput, dtype=torch.float)
y = torch.tensor(yInput)
net = Net(x.shape[1])

mse = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=10e-8)
```

## 训练

```
n_iters = 1000
loss_val = []
for i in range(n_iters):
    yhat = net(x)
    loss = mse(y, yhat)

    optimizer.zero_grad() # 先清除之前的梯度
    loss.backward()     # 计算梯度
    optimizer.step()   # 更新参数

    loss_val.append(loss.item())

plt.plot(loss_val)
plt.show()
print("MSE: ", loss.item())
print("weights: ")
for x in net.parameters():
    print(x)
```

输出

```
MSE:  191.11294272656139
weights: 
Parameter containing:
tensor([[ 0.0299,  0.1689, -0.1578,  0.0841, -0.1880, -0.1485, -0.2007,  0.1927,
          0.1962,  0.0508, -0.0393,  0.0476, -0.2572]], requires_grad=True)
Parameter containing:
tensor([-0.0849], requires_grad=True)
```

![picture](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200428195137.png)

from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import torch

plt.ion() # 开启interactive mode

boston = load_boston()
xInput = boston.data
yInput = boston.target

class Net(torch.nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

x = torch.tensor(xInput, dtype=torch.float)
y = torch.tensor(yInput)
net = Net(x.shape[1])

mse = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=10e-8)

n_iters = 1000
loss_val = []
for i in range(n_iters):
    print("Epoch {}/{}".format(i, n_iters))
    print("-" * 10)

    yhat = net(x)
    loss = mse(y, yhat)

    optimizer.zero_grad() # 先清除之前的梯度
    loss.backward()     # 计算梯度
    optimizer.step()   # 更新参数

    loss_val.append(loss.item())
    print("Train loss: {}".format(loss.item()), end="\n\n")

plt.plot(loss_val)
# plt.savefig('picture name') # 保存图片
# plt.show(block=False)
plt.pause(0.001) # 显示秒数
plt.close() # 关闭图片，并继续执行后续代码。
print("MSE: ", loss.item())
print("weights: ")
for x in net.parameters():
    print(x)

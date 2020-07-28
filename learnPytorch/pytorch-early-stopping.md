[toc]

# Pytorch Early Stopping

## 实现

Pytorch 中没有实现 Early Stopping 的方法。可以自定义下面的一个类，用这个类来进行 Early Stopping。这个类来源于 [https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb](https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb)。

### 定义

```
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

```


### 使用

```
model = yourModel()
criterion = nn.CrossEntropyLoss()	
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
early_stopping = EarlyStopping(patience=20, verbose=True)

batch_size = 64
n_epochs = 100
#----------------------------------------------------------------
training_dataset = Data.TensorDataset(X_train, y_train)
data_loader = Data.DataLoader(
    dataset=training_dataset,
    batch_size=batch_size,	# 批量大小
    shuffle=True	# 是否打乱数据顺序
)

# 训练模型，直到 epoch == n_epochs 或者触发 early_stopping 结束训练
for epoch in range(1, n_epochs + 1):

    #---------------------------------------------------
    model.train()
	for batch, (data, target) in enumerate(data_loader):
		output = model(data)
		loss = criterion(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	#----------------------------------------------------

	model.eval() # 设置模型为评估/测试模式
	valid_output = model(X_val)
	valid_loss = criterion(valid_output, y_val)

	early_stopping(valid_loss, model)
	if early_stopping.early_stop:
		print("Early stopping")
		break

# 获得 early stopping 时的模型参数
model.load_state_dict(torch.load('checkpoint.pt'))
```

# References
1. [在 Pytorch 中实现 early stopping_人工智能_qq_37430422的博客-CSDN博客](https://blog.csdn.net/qq_37430422/article/details/103638681?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase)
2. [https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb](https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb)。

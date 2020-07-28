[toc]

# Pytorch Loss

## torch.nn.CrossEntropyLoss

主要记得两点：
1. 第一个参数是 logits，也就是不需要过 softmax 函数。
2. 第二个参数是不是one_hot形式的，而且其类型应该为 `torch.long`

使用

>The input is expected to contain raw, unnormalized scores for each class.

```
import torch.nn
import torch.nn as nn

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
```

# References
1. [torch.nn — PyTorch master documentation](https://pytorch.org/docs/stable/nn.html#crossentropyloss)

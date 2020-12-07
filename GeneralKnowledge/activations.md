
[toc]

# 学习笔记 - 11 激活函数

## 为什么需要激活函数？

神经元本层的输出在作为下一层的输入之前，需要先进行一个函数变换。这个函数就是激活函数。

激活函数的主要作用是添加非线性，是神经网络非线性的来源。如果没有激活函数的话，即使堆加多层神经网络，也只能起到线性变换的效果。此时根据 loss 的不同，神经网络会退化为线性回归或 Logistic 回归。

## 不同的激活函数比较

### sigmoid

$$
\sigma(x) = \frac{1}{1 + \exp^{-x}}
$$ 

优点：
1. 输出值为 0、1 之间，可以做为输出层输出概率
2. 平滑。这一点很重要。只有平滑才可以算导数。
3. 导数比较好计算。为 $\sigma(x) (1-sigma(x))$ 

缺点：
1. 梯度很小。会导致梯度消失
2. 涉及指数运算，计算量大。
3. 导数不是关于0对称的。至于为什么关于不0对称不好，可以看 [2]

$$
\sigma(x)' = \sigma(x) (1 - \sigma(x)) \le (\frac{1}{2})^2 = \frac{1}{4}
$$ 


### tanh

$$
\begin{aligned}
    \sigma(x) &= \frac{e^x - e^{-x}}{e^x + e^{-x}} \\
    \sigma(x)' &=  1 - \sigma^2(x) \\
.\end{aligned}
$$ 
解决了 sigmoid 不是按照 0 对称的问题。其它性质和 sigmoid 类似。

### relu

优点：
1. 容易求导。非零区域实际上是线性的，因此计算代价小
2. 避免了梯度消失问题。
3. 提供了神经网络的稀疏性。

缺点：
1. 会造成 dead relu 问题。如果函数值通过 relu 函数就会一直变成0，导致了神经元的不可逆死亡。这一点可以使用 leaky relu 来解决。

# References
1. [几种常见激活函数(笔记整理)_网络_u010513327的博客-CSDN博客](https://blog.csdn.net/u010513327/article/details/81019522?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3)
2. [谈谈激活函数以零为中心的问题 | 始终](https://liam.page/2018/04/17/zero-centered-active-function/)
3. [(1条消息)浅谈深度学习中的激活函数_绝对不要看眼睛里的郁金香的博客-CSDN博客_relu是最近几年非常受欢迎的激活函数。被定义为y={0x(x≤0)(x>0)y={0(x≤0)x(](https://blog.csdn.net/qq_17754181/article/details/53179483?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.edu_weight&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.edu_weight)

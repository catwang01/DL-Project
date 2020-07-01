[toc]

# RNN Numpy 实现一—— 计算图与公式推导

这篇文章用来手动使用 numpy 实现一个简单的 RNN 神经网络，从细节中可以加深对 RNN 的理解。

## RNN 模型

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200701110720.png)

RNN 的结构如图所示。左侧的是比较简洁的表达，其中黑色的小块可以理解为一个穿越时间的小门，可以 h 在上一时间 t-1 的结果通过这个小门被连接到了当时时刻 t 。

右侧的图片是按时间展开的表示，这个按时间展开的操作被称为 unfold。注意，在这个图中，为了简洁我们省略了 bias，并且将权重weight都画在了边上。实际上，weight和bias都是一个节点。如下面的图所示

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200701112133.png)

这个图是完整的计算图。但是有点太复杂了。这里画出来只是加深理解。

### 关于序列长度

注意，上面的图中有一个参数 T 表示序列长度，表示按照时间展开的步数，也即序列的长度。这个是笔者在刚开始理解 RNN 时没有理解的一个概念，导致当时笔者无法理解 RNN 的输出应该如何组织，这里特殊说明。

假如我们有一句话，我们希望用这句话中的前一个词来预测后一个词，那么 T = 1；如果我们希望用前三个词来预测后三个词，那么 T = 3；

序列长度为什么重要？因为这关系到我们如何组织。还是用上面的例子，假如我们有一句话，用 a b c e d f 来表示，我们希望用前三个词来预测后三个词，那么我们就应该将这句话的每连续三个词看做 X，如 X = [a, b, c]，将这三个词紧接的那个词看做 y，如 y = [d]，这样我们就有一个样本 X, y = ([a,b,c], d)，从一句话中可以得到许多这样的样本来，如从 a b c d e f 这句话中，我们可以得到下面的样本：

| x | y |
| -- | -- |
| a b c | d |
| b c d | e |
| c d e | f |

可以看到，RNN 实际上只能处理定长的序列，不过可以通过一些方式来处理变长序列。这里就不叙述了。

## 正向传播

由计算图，可以得到正向传播的公式为：

$$
\begin{aligned}
    h_t &= tanh( W_{xh} x_t + W_{hh} h_{t-1} + b_{h} ) \\
    \hat{y}_t &= softmax(W_{hy} h_t + b_y) \\
    L_t &= crossentropy(y_t, \hat{y}_t) \\
\end{aligned}
$$ 

其中 $L_t$ 只是对于 $t$ 时刻的损失，整个序列的损失为

$$
L = \frac{1}{T} \sum_{t=1}^{T} L_t
$$ 

为了更好的理解网络的结构我们添加一些中间变量，

$$
\begin{aligned}
    a_t &=  W_{xh} x_t + W_{hh} h_{t-1} + b_{h}  \\
    h_t &= tanh(a_t) \\
    o_t &= W_{hy} h_t + b_y \\
    \hat{y}_t &= softmax(o_t)  = \frac{exp(o_t)}{1^T exp(o_t)}\\
    L_t &= crossentropy( y_t, \hat{y}_t)  \\
    &= 1^T ( y_t \odot \log \hat{y}_t)\\
    &= y_t ^T \log \hat{y}_t \\
\end{aligned}
$$ 

这里，RNN 的结构可以表示为下面的图片

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200701115337.png)


在这张图片里，我们不仅将中间结果的节点画了出来，还画出了损失函数的节点，还标出了 bias。

# Appendix RNN 反向传播公式的推导

有 t 个时刻的反向传播的结果需要计算，并且反向传播的梯度是从 t+1 时刻传播到 t 时刻。

我们假设已经计算了 $t+1, t+2, \ldots, T$ 时间的反向传播的结果，接下来我们来推导 $t$ 时刻的反向传播的结果。

注意： 这里使用矩阵的微分和导数的关系结合矩阵的迹来计算导数。有关矩阵的微分和导数的内容，可以参考 [ 1 ]

由于 $L = \frac{1}{T} \sum_{t=1}^{T} L_t$  ， 为了化简表达，我们可以先计算 $L = \sum_{t=1}^{T}  L_t$  的结果，最后再除以 $T$ 。

## 1. $L \to L_t$  

$$\frac{\partial L}{\partial L_t} = 1$$ 

## 2. $L_t \to  \hat{y}_t$ 

利用微分
$$
\begin{aligned}
    dL_t &= - d(1^T (y_t \odot log \hat{y}_t)) \\
    &= - d(y_t ^T  log \hat{y}_t)  \\
    &= - y_t^T d log \hat{y}_t \\
    &= -  y_t^T \left( \frac{1}{ \hat{y}_t } \odot d\hat{y}_t\right) \\
    &= -  tr\left( y_t^T \left( \frac{1}{ \hat{y}_t } \odot d\hat{y}_t\right) \right) \\
    &= -  tr\left( \left(y_t \odot \frac{1}{ \hat{y}_t} \right)^T d \hat{y}_t\right ) \\
.\end{aligned}
$$ 

由此得到导数

$$
\frac{\partial L}{\partial \hat{y}_t} = - y_t \odot \frac{1}{ \hat{y}_t }
$$ 

## 3. $\hat{y}_t \to o_t$ 

$$
\begin{aligned}
    d \hat{y}_t &=  d softmax(o_t) \\
    &=  d  \frac{\exp (o_t)}{ 1^T \exp (o_t)} \\
    &= \frac{1}{ 1^T \exp(o_t) } d \exp(o_t)  + \exp(o_t) \frac{-1}{\left(1^T \exp(o_t)\right)^2} d(1^T \exp(o_t))\\
    &= \frac{1}{ 1^T \exp(o_t) } \left( \exp(o_t) \odot d o_t \right)  + \exp(o_t) \frac{-1}{\left(1^T \exp(o_t)\right)^2} 1^T d(\exp(o_t)) \\
    &= \hat{y}_t \odot d o_t  + \exp(o_t) \frac{-1}{\left(1^T \exp(o_t)\right)^2} 1^T (\exp(o_t) \odot do_t) \\
    &= \hat{y}_t \odot d o_t  + \exp(o_t) \frac{-1}{\left(1^T \exp(o_t)\right)^2} \exp(o_t)^T do_t \\
    &= \hat{y}_t \odot d o_t  - \hat{y}_t \hat{y}_t^T do_t \\
    &= \hat{y}_t \odot \left( d o_t  -  1 \hat{y}_t^T do_t \right)  \\
    &= \hat{y}_t \odot \left( \left( I -  1 \hat{y}_t^T \right) do_t \right)  \\
.\end{aligned}
$$ 

因此，

$$
\begin{aligned}
   dL &= tr\left( \left(\frac{\partial L}{\partial \hat{y}_t} \right)^T d \hat{y}_t \right) \\ 
   &= tr\left( \left(\frac{\partial L}{\partial \hat{y}_t} \right)^T \left( \hat{y}_t \odot \left( \left( I -  1 \hat{y}_t^T \right) do_t \right) \right) \right) \\ 
   &= tr\left( \left( \frac{\partial L}{\partial \hat{y}_t} \odot \hat{y}_t \right)^T \left( I - 1 \hat{y}_t^T \right) do_t   \right)  \\
   &= tr\left( \left(- y_t \odot \frac{1}{ \hat{y}_t} \odot \hat{y}_t \right)^T \left( I - 1 \hat{y}_t^T \right) do_t   \right)  \\
   &= tr\left(- y_t^T \left( I - \mathbf{1} \hat{y}_t^T \right) do_t   \right)  \\
   &= tr\left(  y_t^T \mathbf{1} \hat{y}_t^T  do_t - y_t^T do_t   \right)  \\
   &= tr\left( \hat{y}_t^T do_t - y_t^T do_t  \right)  (根据事实 y_t^T \mathbf{1} = 1) \\
   &= tr\left( \left( \hat{y}_t - y_t  \right)^T do_t \right)  \\
.\end{aligned}
$$ 

由此可以得到，

$$
\frac{\partial L}{\partial o_t} = \hat{y}_t - y_t
$$ 

## 4.  $o_t \to h_t$ 

需要注意的是，$h_t$ 在计算图中有两个父节点  $o_t$  和 $a_{t+1}$ ，因此在计算  $\frac{\partial L}{\partial h_t}$ 时需要考虑这两部分。

### $\frac{\partial L}{\partial h_t}$  的计算

$$
do_t = d \left( W_{hy} h_t + b_y \right) = W_{hy} dh_t \left( 这里只考虑了与 dh_t 相关的内容\right) 
$$ 

因此，

$$
\begin{aligned}
dL &= tr\left( \left( \frac{\partial L}{\partial o_t} \right)^T do_t \right)  \\
&= tr\left( \left( \frac{\partial L}{\partial o_t} \right)^T W_{hy} dh_t \right)  \\
.\end{aligned}
$$ 

得

$$
\begin{aligned}
\frac{\partial L}{\partial h_{t1}} &= W_{hy}^T \frac{\partial L}{\partial o_t} \\
&= W_{hy}^T \left( \hat{y}_t - y_t \right)   \\
.\end{aligned}
$$ 

考虑 $a_{t+1}$ 流向  $h_t$ 的梯度，由于 $t+1$ 时刻的梯度先于  $t$ 时刻的梯度被计算出来，因此此时  $\frac{\partial L}{\partial a_{t+1}}$  的结果是已知的，可以直接使用。根据矩阵微分和导数之间的关系，有

$$
dL = tr\left( \left( \frac{\partial L}{\partial a_{t+1}} \right) ^T da_{t+1}  \right) 
$$ 

而 $a_{t+1} = W_{xh} x_{t+1} + W_{hh} h_t + b_h$ 

$$
da_{t+1} = W_{hh} dh_t （忽略和W_{hh}无关的项）
$$ 

$$
\begin{aligned}
dL &= tr\left( \left( \frac{\partial L}{\partial a_{t+1}} \right)  ^T  W_{hh} dh_t\right)  \\    
.\end{aligned}
$$ 

因此可以得到，

$$
\frac{\partial L}{\partial h_{t2}} = W_{hh}^T \frac{\partial L}{\partial a_{t+1}}
$$ 

将两部分加和，有

$$
\begin{aligned}
    \frac{\partial L}{\partial h_t}  &= \frac{\partial L}{\partial h_{t1}} + \frac{\partial L}{\partial h_{t2}} \\
    &= W_{hy}^T \left( \hat{y}_t - y_t \right) + W_{hh}^T \frac{\partial L}{\partial a_{t+1}}
.\end{aligned}
$$ 

### $\frac{\partial L}{\partial W_{hy}^{t}}$  的计算

注意，由 $L$ 到  $W_{hy}$ 的路径有 $T$ 条，每一条都会有梯度流向 $W_{hy}$，$L$ 对于 $W_{hy}$ 的梯度是这 $T$ 条路径流向 $W_{hy}$ 的梯度之和。
为了区别这 $T$ 个梯度， 用  $W_{hy}^t$ 来表示 $t$ 时刻对应的路径流向  $W_{hy}$ 的梯度。$W_{hh}$ 、$W_{xh}$ 、$b_y$, $b_h$   也有类似的表示

先计算 $do_t$ 

$$
do_t = d \left( W_{hy} h_t + b_y \right) = dW_{hy} h_t \left( 这里只考虑了与 dW_{hy} 相关的内容\right) 
$$ 

将 $do_t$  代入

$$
\begin{aligned}
dL &= tr \left( \left( \frac{\partial L}{\partial o_t} \right)^T do_t \right)   \\        
&= tr \left( \left( \frac{\partial L}{\partial o_t} \right)^T dW_{hy}h_t\right)   \\        
&= tr \left( h_t \left( \frac{\partial L}{\partial o_t} \right)^T dW_{hy}\right)   \\        
.\end{aligned}
$$ 

由微分和导数的关系可以得到，

$$
\begin{aligned}
    \frac{\partial L}{\partial W_{hy}^t} =  \frac{\partial L}{\partial o_t} h_t^T
.\end{aligned}
$$ 

### $\frac{\partial L}{\partial b_y^t}$ 的计算

先计算 $do_t$ 

$$
do_t = d \left( W_{hy} h_t + b_y \right) = d b_y \left( 这里只考虑了与 db_y 相关的内容\right) 
$$ 

将 $do_t$  代入

$$
\begin{aligned}
dL &= tr \left( \left( \frac{\partial L}{\partial o_t} \right)^T do_t \right)   \\        
&= tr \left( \left( \frac{\partial L}{\partial o_t} \right)^T db_y\right)   \\        
.\end{aligned}
$$ 

由微分和导数的关系可以得到，

$$
\frac{\partial L}{\partial b_{y}^t} =  \frac{\partial L}{\partial o_t}
$$ 

## 5. $h_t \to a_t$ 

先计算 $dh_t$ 

$$
\begin{aligned}
    dh_t &= d\left( tanh(a_t) \right)   \\
    &= (1 - tanh(a_t)^2) \odot  da_t \\
    &= \left( 1 - h_t^2 \right)  \odot da_t
.\end{aligned}
$$ 

代入微分公式中

$$
\begin{aligned}
dL &= tr\left( \left( \frac{\partial L}{\partial h_t} \right)^T dh_t  \right)  \\        
&= tr\left( \left( \frac{\partial L}{\partial h_t} \right)^T  \left( 1 - h_t^2 \right)  \odot da_t \right)  \\        
&= tr\left( \left( \frac{\partial L}{\partial h_t} \odot \left( 1 - h_t^2 \right) \right)^T da_t \right)  \\        
.\end{aligned}
$$ 

由微分和导数的关系可得

$$
\begin{aligned}
 \frac{\partial L}{\partial a_t} &= \frac{\partial L}{\partial h_t} \odot (1 - h_t^2)
.\end{aligned}
$$ 

## 6. $a_t \to x_t$ 

### $\frac{\partial L}{\partial W_{xh}^t}$ 的计算

先计算 $da_t$ 

$$
\begin{aligned}
    da_t &= d\left( W_{xh} x_t + W_{hh} h_{t-1} + b_h \right)  \\
    &= dW_{xh} x_t （只保存和 W_{xh} 相关的项）\\
.\end{aligned}
$$ 

代入微分公式中，有

$$
\begin{aligned}
 dL &= tr\left( \left( \frac{\partial L}{\partial a_t} \right)^T da_t  \right)  \\ 
    &= tr\left( \left( \frac{\partial L}{\partial a_t} \right)^T dW_{xh} x_t  \right)  \\ 
    &= tr\left( x_t \left( \frac{\partial L}{\partial a_t} \right)^T dW_{xh}  \right)  \\ 
.\end{aligned}
$$ 

由微分与导数的关系，得到

$$
\frac{\partial L}{\partial W_{xh}^t}  = \frac{\partial L}{\partial a_t}  x_t^T
$$ 

### $\frac{\partial L}{\partial b_h^t}$  的计算

先计算 $da_t$ 

$$
\begin{aligned}
    da_t &= d\left( W_{xh} x_t + W_{hh} h_{t-1} + b_h \right)  \\
    &= db_h（只保存和 b_h 相关的项）\\
.\end{aligned}
$$ 

代入微分公式中，有

$$
\begin{aligned}
 dL &= tr\left( \left( \frac{\partial L}{\partial a_t} \right)^T da_t  \right)  \\ 
    &= tr\left( \left( \frac{\partial L}{\partial a_t} \right)^T db_h  \right)  \\ 
.\end{aligned}
$$ 

由微分与导数的关系，得到

$$
\frac{\partial L}{\partial b_h^t}  = \frac{\partial L}{\partial a_t} 
$$ 


### $\frac{\partial L}{\partial W_{hh}^t}$  的计算

先计算 $da_t$ 

$$
\begin{aligned}
    da_t &= d\left( W_{xh} x_t + W_{hh} h_{t-1} + b_h \right)  \\
    &= dW_{hh} h_{t-1}（只保存和 W_{hh} 相关的项）\\
.\end{aligned}
$$ 

代入微分公式中，有

$$
\begin{aligned}
 dL &= tr\left( \left( \frac{\partial L}{\partial a_t} \right)^T da_t  \right)  \\ 
    &= tr\left( \left( \frac{\partial L}{\partial a_t} \right)^T dW_{hh} h_{t-1}  \right)  \\ 
    &= tr\left( dW_{hh} \left( \frac{\partial L}{\partial a_t} \right)^T  h_{t-1}  \right)  \\ 
.\end{aligned}
$$ 

由微分与导数的关系，得到

$$
\frac{\partial L}{\partial W_{hh}^t}  = \frac{\partial L}{\partial a_t}  W_{hh}^T
$$ 

## 总结

汇总一下上面的结果，有

$$
\begin{aligned}
\frac{\partial L}{\partial W_{hy}^t} &=  \frac{\partial L}{\partial o_t} h_t^T \\
\frac{\partial L}{\partial b_{y}^t} &=  \frac{\partial L}{\partial o_t} \\
\frac{\partial L}{\partial W_{xh}^t}  &= \frac{\partial L}{\partial a_t}  x_t^T \\
\frac{\partial L}{\partial b_h^t}  &= \frac{\partial L}{\partial a_t}  \\
\frac{\partial L}{\partial W_{hh}^t}  &= \frac{\partial L}{\partial a_t}  W_{hh}^T
.\end{aligned}
$$ 

这是 $t$ 时刻的梯度， $L$ 对于这些参数的梯度是各个时刻的梯度之和，由此得

$$
\begin{aligned}
\frac{\partial L}{\partial W_{hy}} &=  \frac{1}{T}\sum_{t=1}^{T} \frac{\partial L}{\partial o_t} h_t^T \\
\frac{\partial L}{\partial b_{y}} &=  \frac{1}{T} \sum_{t=1}^{T} \frac{\partial L}{\partial o_t} \\
\frac{\partial L}{\partial W_{xh}}  &= \frac{1}{T} \sum_{t=1}^{T} \frac{\partial L}{\partial a_t}  x_t^T \\
\frac{\partial L}{\partial b_h}  &= \frac{1}{T} \sum_{t=1}^{T} \frac{\partial L}{\partial a_t}  \\
\frac{\partial L}{\partial W_{hh}}  &= \frac{1}{T} \sum_{t=1}^{T}  \frac{\partial L}{\partial a_t}  W_{hh}^T
.\end{aligned}
$$ 

# References

1. [矩阵求导术（上） - 知乎](https://zhuanlan.zhihu.com/p/24709748)



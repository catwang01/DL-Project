[toc]

# Deep Learning from Scratch Chapter6 与学习相关的技巧

## 6.1 参数更新

### 6.1.2 SGD

SGD 的更新式

$$W \rightarrow W - \eta \frac{\partial L}{\partial W}$$


#### SGD类的实现

```
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
```

其中 params, grads 是字典型变量，按 params['W1']、grads['W1'] 的形式，分别保存了权重参数和梯度。

之后所有的优化器都采取这样的形式

下面是使用SGD类的伪代码

```
network = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000):
    ...
    x_batch, t_batch = get_mini_batch(...) # mini-batch
    grads = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer.update(params, grads)
```

### 6.1.4 Momentum

$$
\begin{aligned}
v \leftarrow \alpha v - \eta \frac{\partial L}{\partial W} 
\\
W \leftarrow W + v
\end{aligned}
$$

v 是和 W 相同大小的矩阵。表达历史速度，刚开始时没有历史速度，因此初始化为 0 

#### Momentum实现

```
class Momentum:
    def __init__(self, alpha=0.9, lr=0.01):
        self.alpha = alpha
        self.lr = lr
        self.v = None
    
    def update(self, params, grads):
        if self.v is None: # 初始化 self.v 为0，更开始没有速度
            self.v = {np.zeros_like(params[key]) for key in params}
        
        for key in params:
            self.v[key] = self.alpha * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
```

### 6.1.5 AdaGrad


$$
\begin{aligned}
h \leftarrow h + \frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W} \\

W \leftarrow W - \eta \frac{1}{\sqrt{h}} \frac{\partial L}{\partial W} 
\end{aligned}
$$

#### AdaGrad实现

```
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {key: np.zeros_like(params[key]) for key in params}
        for key in params:
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)       
```

### 6.1.6 RMSProp（自）

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200428143005.png)

RMSProp 将 AdaGrad 中的 h 由平均值修改为指数滑动平均值。公式如下：


$$
\begin{aligned}
h \leftarrow \beta h + (1-\beta)\frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W} \\
W \leftarrow W - \eta \frac{1}{\sqrt{h}} \frac{\partial L}{\partial W} 
\end{aligned}
$$

#### RMSProp实现


```
class RMSProp:
    def __init__(self, decay_rate=0.99, lr=0.01):
        self.decay_rate = decay_rate
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {key: np.zeros_like(params[key]) for key in params}
        for key in params:
            h[key] = self.decay_rate * h[key] + (1 - self.decay_rate) * grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(h[key] + 1e-7))
        
```

### 6.1.7 Adam

下面的公式来自于原论文

![8aa713b60f00c66df4efba5ab62f1109.png](evernotecid://8E200321-31A9-427B-BECA-CC44235980BC/appyinxiangcom/22483756/ENResource/p12746)

#### Adam实现

```
class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.99):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.iter = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = {key: np.zeros_like(val) for key, val in params}
            self.v = {key: np.zeros_like(val) for key, val in params}
            
        for key in params:
            self.iter += 1
            self.m[key] = self.beta1 * self.m[key] + (1-self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            self.m[key] /= (1 - self.beta1**self.iter)
            self.v[key] /= (1 - self.beta2**self.iter)
            params[key] -= self.lr * self.m[key] / np.sqrt(self.v[key] + 1e-7)
```

## 6.3 Batch Normalization


### 6.3.1 BatchNormalization

![2418175f20eb71bfda625ee261b9210f.png](evernotecid://7C71FA3B-B5E0-41C4-AF1B-AB35446A037A/appyinxiangcom/22483756/ENResource/p11822)

## 6.4 正则化

![981e0824fa6b0b672e76dab40cbb694d.png](evernotecid://8E200321-31A9-427B-BECA-CC44235980BC/appyinxiangcom/22483756/ENResource/p12747)

**所谓"删除"神经元，不需要真的改变权重的形状，在实现上就只是将被删除的神经元对应的权重赋值为 0** 

### Dropout层的实现

Dropout 需要进行 rescale，有两种实现方式，
一种是 training 时不变，testing 时乘以 keep_prob，这个是原始论文中的，被称为 vanilla dropout
一种是 training 时除以 keep_prob，testing 时不变，这个被称为 inverted dropout

至于为什么要进行rescale，一种不严谨的解释是保证输出的期望是固定的。
假设某一层的输出为 x，期望为 $Ex$ 。经过 dropout 之后，输出为 $r \odot x$，那么期望变成了 $pEx$ 。如果，对于 inverted dropout 来说，在 training 时进行 rescale
，相当于输出为 $r \odot x / p$ ，期望仍然为 $Ex$ 。

而对于 vanilla dropout 来说，testing  的时候，由于不使用 dropout，因此 tesing 时的期望为 $Ex$，而 training 时结果的期望为 $pEx$  ，相当于 testing 和 training 时的结果有一个常数关系，导致 training 和 testing 的结果不一致。为了让结果一致，需要对 testing 的结果乘以 p，此时 testing 的结果的期望为 $pEx$ 和 training 保持一致。

#### 实现1 -- vanilla dropout

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200729113630.png)


#### 实现2 -- inverted dropout

```
import numpy as np

class Dropout:
    def __init__(self, keep_prob=0.9):
        self.keep_prob = keep_prob
        self.mask = None
        
    def forward(self, x, is_train=True):
        self.mask = np.random.rand(*x.shape) < self.keep_prob
        if is_train:
            return self.mask * x / self.keep_prob
        else:
            return x
    
    def backward(self, dout):
        return self.mask * dout / self.keep_prob
```

这里有一点需要注意。一个batch的数据中有 batch_size 个样本（这里 取batch_sizs=5），删除神经元是对每个样本分别删除，而不是一起删除。

假设输入为

```
[[5 8 9 5]
 [0 0 1 7]
 [6 9 2 4]
 [5 2 4 2]
 [4 7 7 9]]
```

那么输出可能是 

```
[[0 8 9 0]
 [0 0 0 7]
 [6 9 0 0]
 [0 0 0 0]
 [0 7 7 9]]
```

不太可能是 

```
[[0 8 9 0]
 [0 0 1 0]
 [0 9 2 0]
 [0 2 4 0]
 [0 7 7 0]]
```

体现在代码上，就是下面这句
```
self.mask = np.random.rand(*x.shape) < self.keep_prob
# 不是 self.mask = np.random.rand(*x.shape[1]) < self.keep_prob
```

#### Dropout类的测试

测试Dropout类，并和Tensorflow中的结果进行比较

```
import numpy as np
from common.layers import Dropout

x_train, x_test = np.ones((2,3)), np.random.randn(1,3)
keep_prob = 0.4

def testDropout():
    dropout = Dropout(keep_prob)
    y_train= dropout.forward(x_train)
    grad = dropout.backward(np.ones_like(x_train))
    y_test= dropout.forward(x_test, is_train=False)
    return y_train, grad, y_test

def testTFDropout():
    # 使用tensorflow的dropout来进行对比
    import tensorflow as tf
    from tensorflow.contrib.layers import dropout
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool)
    x = tf.placeholder(tf.float32, shape=(None, 3))
    y = dropout(x, keep_prob, is_training=is_training)
    grad = tf.gradients(y, [x])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        y_train, grad = sess.run([y, grad], 
                                 feed_dict={x: x_train, is_training: True}
                                )
        y_test = sess.run(y, feed_dict={x: x_test, is_training: False})
    return y_train, grad, y_test

if __name__=="__main__":
    y_train1, dx1, y_test1 = testDropout()
    y_train2, dx2, y_test2 = testTFDropout()
    
    print("y_train:", y_train)
    print("y_train:", y_train)

    print("dx1:", grad)
    print("dx2:", grad)

    print("ytrain1:", y_test)
    print("ytrain2:", y_test)
```


# References

1. [(1 封私信 / 28 条消息) 神经网络Dropout层中为什么dropout后还需要进行rescale？ - 知乎](https://www.zhihu.com/question/61751133/answer/190722593)

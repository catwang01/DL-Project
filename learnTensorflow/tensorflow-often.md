[toc]

# Tensorflow2 常用函数

## 安装

```
pip install tensorflow
```

如果是使用 conda 虚拟环境，一定不能使用 pip3，而是要用 pip

```
print(tf.constant([1,2,3]))
```

tensorflow2 将静态图修改为动态图。在 tensorflow1中，下面的语句的返回为

```
# tensorflow1
<tf.Tensor 'Const:0' shape=(3,) dtype=int32>
# tensorflow2
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
```


## 常用函数

### 一些 general的东西

标量的 shape=() 

### 随机数相关

#### tf.random.set_seed 设置种子

```
import tensorflow as tf

tf.random.set_seed(12345)
print(tf.random.normal(())) # 0.273780
print(tf.random.normal(())) # 0.21987937

tf.random.set_seed(12345)
print(tf.random.normal(())) # 0.2737803
print(tf.random.normal(())) # 0.21987937
```

### .dtype 得到类型

### .numpy() 得到numpy对象

### tf.cast

```
x1 = tf.constant([[1.0, 2.0],
                 [2.0, 3.0]])    
y_cast = tf.cast(x1, tf.int64)
print(y_cast)
```

结果

```
tf.Tensor(
[[1 2]
 [2 3]], shape=(2, 2), dtype=int64)
```

### tf.reduce_min, tf.reduce_max, tf.reduce_mean, tf.reduce_sum

```
x1 = tf.constant([[1.0, 4.0],
                 [3.0, 2.0]])    

y_min = tf.reduce_min(x1) # tf.Tensor(1.0, shape=(), dtype=float32)

# 对行计算
y_max_row = tf.reduce_max(x1, axis=1) # <tf.Tensor: shape=(3,), dtype=float32, numpy=array([4., 3., 6.], dtype=float32)>

# 保持维度 keepdims=True
tf.reduce_sum(x1, keepdims=True) # <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[20.]], dtype=float32)>
```


### tf.convert_to_tensor

```
import numpy as np

x = np.array([1,2,3])
tf.convert_to_tensor(x) # <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 2, 3])>
```

### tf.constant

和 tf.convert_to_tensor 的用法相同，二者使用一个就可。

### tf.data.Dataset.from_tensor_slices

```
features = tf.constant([12, 23, 10])
labels = tf.constant([0, 1, 1])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for x, y in dataset:
    print("x:", x, "y:", y)
```

结果：

```
x: tf.Tensor(12, shape=(), dtype=int32) y: tf.Tensor(0, shape=(), dtype=int32)
x: tf.Tensor(23, shape=(), dtype=int32) y: tf.Tensor(1, shape=(), dtype=int32)
x: tf.Tensor(10, shape=(), dtype=int32) y: tf.Tensor(1, shape=(), dtype=int32)
```
    
### tf.GradientTape

```
with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = w ** 2 #loss=w2 loss’=2w
grad = tape.gradient(loss, w) # tf.Tensor(6.0, shape=(), dtype=float32)
```

### tf.one_hot

```
classes = 3
labels=tf.constant([1,0,2]) #输入的元素值最小为0，最大为2 
output = tf.one_hot( labels, depth=classes)
print(output)
```

结果

```
tf.Tensor(
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]], shape=(3, 3), dtype=float32)
```

### tf.nn.softmax

```
y = tf.constant ( [1.01, 2.01, -0.66] )
y_pro = tf.nn.softmax(y) # tf.Tensor([0.25598174 0.69583046 0.04818781], shape=(3,), dtype=float32)
```

### assign_sub

用于 Variable 进行自减操作。会返回自减后的 w，同时也会修改 w

```
w = tf.Variable(4)
# 相当于 w -= 1
w.assign_sub(1) # <tf.Variable 'UnreadVariable' shape=() dtype=int32, numpy=3>
```

### argmax / argmin

```
import numpy as np
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])

tf.argmax(test, axis=0) # <tf.Tensor: shape=(3,), dtype=int64, numpy=array([3, 3, 1])>
tf.argmin(test, axis=1) # <tf.Tensor: shape=(4,), dtype=int64, numpy=array([0, 0, 2, 2])>
```


![c3989ae3f13e2dd7d0dd0efb76c72272.png](evernotecid://7E3AE0DC-DC71-4DDC-9CC8-0C832D6C11C2/appyinxiangcom/22483756/ENResource/p11787)




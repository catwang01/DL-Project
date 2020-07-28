[toc]

#  Tensorflow2 GradientTape 的使用

## 基本使用

对于 tf.constant 创建的变量，要计算，需要 watch；如果不 watch，计算出来的是 None

```
import tensorflow as tf
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    y = x * x
dx = tape.gradient(y, x)
print(dx) # None
```

```
import tensorflow as tf
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x
dx = tape.gradient(y, x)
print(dx) # tf.Tensor(6.0, shape=(), dtype=float32)
```

对于 tf.Variable 创建出来的，会自动加入求导列表中，无需 watch

```
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x * x
dx = tape.gradient(y, x)
print(dx) # tf.Tensor(6.0, shape=(), dtype=float32)
```

## Persistent 参数

默认情况下，tape 只能用来计算一次导数。第二次计算导数会报错

```
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x * x
dx = tape.gradient(y, x)
new_dx = tape.gradient(y, x)
print(dx)
print(new_dx) # GradientTape.gradient can only be called once on non-persistent tapes
```

使用 persistent=True 可以让 tape 保持导数，可以多次计算。

注意： 这时需要手动 GC。

```
x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    y = x * x
dx = tape.gradient(y, x)
new_dx = tape.gradient(y, x)
print(dx)   # tf.Tensor(6.0, shape=(), dtype=float32)
print(new_dx) # tf.Tensor(6.0, shape=(), dtype=float32)
del tape # 手动 GC
```

## 计和高阶导数

```
x = tf.Variable(3.0)

with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * x
    dx = t2.gradient(y, x)
ddx = t1.gradient(dx, x)

print(dx) 
print(ddx) # GradientTape.gradient can only be called once on non-persistent tapes
```

# References
1. [GradientTape](https://www.loner.net.cn/gradienttape/)

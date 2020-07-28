[toc]

# Pytorch数据类型转换

Pytorch的数据类型为各式各样的Tensor,Tensor可以理解为高维矩阵。与Numpy中的Array类似。Pytorch中的tensor又包括CPU上的数据类型和GPU上的数据类型，一般GPU上的Tensor是CPU上的Tensor加cuda()函数得到。通过使用Type函数可以查看变量类型。一般系统默认是torch.FloatTensor类型。例如data = torch.Tensor(2,3)是一个2*3的张量，类型为FloatTensor; data.cuda()就转换为GPU的张量类型，torch.cuda.FloatTensor类型。

下面简单介绍一下Pytorch中变量之间的相互转换。

Tensor 不同类型之间的转换

一般只要在Tensor后加long(), int(), double(),float(),byte()等函数就能将Tensor进行类型转换；

例如：Torch.LongTensor--->Torch.FloatTensor, 直接使用data.float()即可

还可以使用type()函数，data为Tensor数据类型，data.type()为给出data的类型，如果使用data.type(torch.FloatTensor)则强制转换为torch.FloatTensor类型张量。

当你不知道要转换为什么类型时，但需要求a1,a2两个张量的乘积，可以使用a1.type_as(a2)将a1转换为a2同类型。

（2）CPU张量 ---->  GPU张量, 使用data.cuda()

（3）GPU张量 ----> CPU张量 使用data.cpu()

（4）Variable变量转换成普通的Tensor，其实可以理解Variable为一个Wrapper，里头的data就是Tensor. 如果Var是Variable变量，使用Var.data获得Tensor变量

（5）Tensor与Numpy Array之间的转换

Tensor---->Numpy  可以使用 data.numpy()，data为Tensor变量

Numpy ----> Tensor 可以使用torch.from_numpy(data)，data为numpy变量

# References
1. [Pytorch变量类型转换_人工智能_zchenack个人专栏-CSDN博客](https://blog.csdn.net/hustchenze/article/details/79154139)

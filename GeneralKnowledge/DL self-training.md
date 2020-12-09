[toc]

# DL self-training

## 半监督学习

半监督学习是一种介于监督式学习和无监督学习之间的学习范式，我们都知道，在监督式学习中，样本的类别标签都是已知的，学习的目的找到样本的特征与类别标签之间的联系。一般来讲训练样本的数量越多，训练得到的分类器的分类精度也会越高。

但是在很多现实问题当中，一方面由于人工标记样本的成本十分高昂，导致了有标签的样本十分稀少。而另一方面，无标签的样本很容易被收集到，其数量往往是有标签样本的上百倍。半监督学习（这里仅针对半监督分类）就是要利用大量的无标签样本和少量的有标签样本来训练分类器，解决有标签样本不足这个难题。

## self-training

`self-training` 可能是最早被提出来的半监督学习方法，最早可以追溯到Scudder(1965)。`self-training` 相比其它的半监督学习方法的优势在于简单以及不需要任何假设。

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20201208172340.png)


上面是self-training的算法流程图，简单解释一下：

1. 将初始的有标签数据集作为初始的训练集 $\left(X_{\text {train}}, y_{\text {train}}\right)=\left(X_{l}, y_{l}\right)$ 根据训练集训练得到一个初始分类器 $C_{int}$。

2. 利用 $C_{int}$ 对无标签数据集 $X_u$ 中的样本进行分类，选出最有把握的样本 $\left(X_{\text {conf}}, y_{\text {conf}}\right)$

3. 从 $X_u$ 中去掉 $\left(X_{\text {conf}}, y_{\text {conf}}\right)$

4. 将 $\left(X_{\text {conf}}, y_{\text {conf}}\right)$ 加入到有标签数据集中，

$$
\left(X_{\text {train}}, y_{\text {train}}\right) \leftarrow\left(X_{l}, y_{l}\right) \cup\left(X_{\text {conf}}, y_{\text {conf}}\right)
$$ 

5. 根据新的训练集训练新的分类器，重复步骤2到5直到满足停止条件（例如所有无标签样本都被标记完了）

6. 最后得到的分类器就是最终的分类器。


# References
1. [(1条消息) 半监督学习之self-training_tyh70537的博客-CSDN博客](https://blog.csdn.net/tyh70537/article/details/80244490)
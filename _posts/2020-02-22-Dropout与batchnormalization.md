---
layout:     post
title:      Dropout与batchnormalization
subtitle:   Dropout与batchnormalization
date:       2020-02-22
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 神经网络
---

### Dropout

https://zhuanlan.zhihu.com/p/38200980 (部分有错)
https://zhuanlan.zhihu.com/p/61725100

- 训练时还要对第二层的输出数据除以（1-p）之后再传给输出层神经元，作为神经元失活的补偿，以使得在训练时和测试时每一层输入有大致相同的期望。
- dropout只在训练时，预测时不进行
- 

### BN

#### Internal Covariate Shift(我们为什么需要BN)
我们知道网络一旦train起来，那么参数就要发生更新，除了输入层的数据外(因为输入层数据，我们已经人为的为每个样本归一化)，后面网络每一层的输入数据分布是一直在发生变化的，因为在训练的时候，前面层训练参数的更新将导致后面层输入数据分布的变化。以网络第二层为例：网络的第二层输入，是由第一层的参数和input计算得到的，而第一层的参数在整个训练过程中一直在变化，因此必然会引起后面每一层输入数据分布的改变。我们把网络中间层在训练过程中，数据分布的改变称之为：“Internal Covariate Shift”。BN的提出，就是要解决在训练过程中，中间层数据分布发生改变的情况。（来自参考1）

####  Internal Covariate Shift会带来什么问题？（来自参考4）
- 上层网络需要不停调整来适应输入数据分布的变化，导致网络学习速度的降低
- 网络的训练过程容易陷入梯度饱和区，减缓网络收敛速度

#### BN步骤
参考4

#### BN的好处
1. 减少内部协变量偏移(Internal Covariate Shift)
2. BN使得网络中每层输入数据的分布相对稳定，加速模型学习速度
3. BN允许网络使用饱和性激活函数（例如sigmoid，tanh等），缓解梯度消失问题
4. BN具有一定的正则化效果
5. 。。。

#### 激活函数/BN/dropout顺序问题
https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras


#### 参考
1. https://www.jianshu.com/p/a78470f521dd
2. https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0
3. https://www.zhihu.com/question/38102762
4. https://zhuanlan.zhihu.com/p/34879333


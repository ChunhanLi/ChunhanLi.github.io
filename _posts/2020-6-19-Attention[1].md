---
layout:     post
title:      Attention[1]
subtitle:   
date:       2020-6-19
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 深度学习
---

### 前言
本来就一直想入门NLP,一直没有机会,刚好有个腾讯广告算法大赛，说是广告大赛，没想到可以算一个NLP比赛，刚好借此机会入门NLP，这篇文章就大概记录下学习Attention的过程。

### 学习资料
1. 介绍attention https://zhuanlan.zhihu.com/p/137578323

这篇知乎应该是入门者最好的选择了,Attention为注意力机制，就像人们看一张图片一样，对图片的各个部位的关注度不同。所以针对每个输入,其都有对应的key,value,query,利用query和key计算出score(关注度),从而给其value加一个权重. 如文中所说,有三个输入,每个输入长度为4,如何去决定每个输入的key,value,query呢？ 这时就需要三个转化矩阵$W_k$,$W_v$,$W_q$,该矩阵的权重网络能够自己学习, 文中矩阵维度为4x3,那么即可将每个输入都转化成3维的key,value,query. 之后,每个query分别和所有的Key计算点积，然后除以根号维度,过softmax,得到score(文中介绍的是点积),加权相加得到输出。 Attention的优点1解决了LSTM/GRU等时间t依赖时间t-1的计算,能够并行,2.能解决LSTM不能彻底解决的的长期依赖现象（任意两个单词的距离是1）

2. 介绍Transformer https://zhuanlan.zhihu.com/p/48508221

Transformer本质是一个Encoder-Decoder的结构，Encoder部分由6个编码器block组成,同样解码器是6个解码block组成,编码器的输出会作为解码器的输入。 每个编码器由一个self-attention和Feed forward Neural Network组成。这个全连接层有两层,第一层的激活函数是relu，第二层是一个线性激活函数,可以表示为$FNN(Z) = \max (0,ZW_1+b_1)W_2+b_2$. Decoder部分比Encoder部分多了一个Encoder-Decoder Attenntion。在self-attention需要强调一点采用了残差网络(输入到第一个add norm(跳过self-attention)|和第二个add norm 跳过FFN)

---
layout:     post
title:      PCA、SVD、LDA
subtitle:   
date:       2020-08-27
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 机器学习
---


#### PCA（最大投影方差角度|最大可分性）

- 矩阵相乘-右边矩阵的每一列向量变换到左边矩阵以每一行行向量为基底所表示的空间里去
- AB = C C中的每一列为新坐标;如果A中基的数量小于B的维度即可以起到降维的作用
- 那么怎么样的降维才算最优的降维？投影后每个基上方差足够大(投影值尽可能分散)
- 一维变量用方差来表示数据的分散程度，多维变量用协方差表示
- 为了使降维完的数据尽量不冗余，协方差为0？不相关 每个方向上方差尽可能大
- 高维变量中，我们用协方差去衡量相关性
- 降维问题的优化目标：将一组 N 维向量降为 K 维，其目标是选择 K 个单位正交基（不同主成分之间没有冗余信息），使得原始数据变换到这组基上后，各变量两两间协方差为 0，而变量方差则尽可能大（在正交的约束下，取最大的 K 个方差）。
- X pxn 维度 协方差矩阵 $\frac{1}{m} X X^T$
- 假设变化完的Y=PX，则Y为X对P做基变换完的数据,Y的协方差矩阵是D，X的协方差矩阵是C
$$D = \frac{1}{m}YY^T = P(\frac{1}{m}XX^T)P^T = PCP^T$$
- 不同特征根对应的特征向量 以他们为列向量构成的正交矩阵P,则$P^{-1}AP = $对角矩阵
- 使得非对角元素为0，对角元素从大到小排序
- 有n个p维数据中心化矩阵$X_{p,n}$，$C = \frac{1}{n}XX^T$为其协方差矩阵
- 设Y = QX，Y的协方差矩阵$D = \frac{1}{n}(QX)(QX)^T = QCQ^T$
- 我们取$Q = P^T$即可

- 优化目标: $$\max_P tr(PCP^T),s.t. PP^T = I$$
用拉格朗日乘子法求解可推出特征根的定义
#### PCA（最小投影距离|最近重构性）
- 投影完且降维完的点到原来点的距离最小
- 也是用拉格朗日乘子法


#### 利用正交矩阵将对称矩阵对角化的步骤
1. 求对称矩阵A的特征值$\lambda_1,\dots,\lambda_n$
2. 对每个特征根,求出其对应的单位化特征向量$P_1,P_2,\dots,P_n$
3. 写出正交矩阵$P = (P_1,P_2,P_3,\dots,P_n)$,则$P^TAP$为对角矩阵


#### PCA基本步骤
1. 对所有样本中心化
2. 求样本协方差矩阵
3. 特征根分解
4. 取前k大的特征根及其对应的特征向量
5. $Y = P^TX$

#### PCA
优点：
1. 降维
2. 降噪
3. 特征之间正交

缺点：
1. 也可能损失一部分特征

#### SVD
- SVD不需要方阵
- $A = U\Sigma V^T$ \
- 其余的看刘建平老师博客

#### LDA
- LDA的基本思想：给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点中心尽可能远离。更简单的概括为一句话，就是“投影后类内方差最小，类间方差最大”。
- 记住瑞利商和广义瑞利商
- 所有类别距离之和$u^TS_bu,S_b$是称为类间散度矩阵
- 各个类别的样本方差之和：$u^TS_wu,S_w$称为类内散度矩阵
- max$\frac{u^TS_bu}{u^TS_wu}$

#### LDA步骤
- 计算$S_w,S_b$
- 计算矩阵$S_w^{-1}S_b$
- 得到它的特征根和特征向量，取最大的K个
- $\omega = (\omega_1,\dots,\omega_k)$
- 得到新样本$z_i = W^Tx_i$

#### LDA和PCA
相同点
- 降维
- 利用了特征根分解
不同点
- LDA有监督/pca无监督
- K个类别 LDA最多降到K-1维度，PCA无限制
#### 参考
- https://zhuanlan.zhihu.com/p/77151308
- https://wenku.baidu.com/view/289210c558f5f61fb736667f.html
- https://www.cnblogs.com/pinard/p/6251584.html
- https://datawhalechina.github.io/pumpkin-book/#/chapter10/chapter10
- https://www.cnblogs.com/pinard/p/6239403.html#!comments
- https://www.cnblogs.com/pinard/p/6244265.html （LDA）
- https://zhuanlan.zhihu.com/p/79696530 （LDA）

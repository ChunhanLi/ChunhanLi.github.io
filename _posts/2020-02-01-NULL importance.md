---
layout:     post
title:      Null importance/Permutation test
subtitle:   
date:       2020-02-01
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 统计机器学习
---
### 前言
在Kaggle 2019 Data Science Bowl比赛中，第一名用了Null importance特征选择方法从20000+特征中筛选出了500个特征。因此，就研究一下Null importance与permutation test特征选择方法。参考[Null importance](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances).
### Null importance
基本步骤
- 正常拟合模型，得到特征重要性。（此处得到的特征重要性为一个基准）
- shuffle 多次target，拟合模型，记录每次得到的特征重要性
- 计算一些比例(例如actual/null mean,acutal/null max)
- 选择靠前的特征，选多少个特征的阈值可自己根据CV调节

### Permutation test
基本步骤
- 正常拟合模型，得到local CV
- 在训练集正常训练，在拟合测试集时shuffle某个特征列的值，记录下其对local CV的影响
- 对每一个特征重复步骤2
- 对于shuffle之后 local CV还能提高的特征 给予删除(阈值可以自己调节，对于那些对local CV降的不多的也可以删除)

### 比较
#### Null importance 
1.省时间,只需要多次拟合模型的时间
2.如果local cv没那么准的时候，null importance也可以生效，因为其考虑的只是特征重要性

#### Permutation test
1.优点...暂时想不出来

当然，具体实战中，哪种方法比较有效，这都是需要去测试的。之前比赛中我使用Permutation test比较多，在前两场比赛中还是生效的，在这个DSB比赛中，permutatition test没起作用，我估摸着一点原因是因为没有好的local cv.
---
layout:     post
title:      Speed uo Python
subtitle:   Python
date:       2018-11-24
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 编程语言
---

[toc]

#### 求平方和的开方


- array中np.linalg.norm(axis=1) 优于 np.sqrt(np.sum(np.power((********),2),axis=1))


#### list转换成array
- [[1,2],[3,4].........] 转换成array 没有 [1,2,3,4,.....] + reshape((-1,2))快

#### preallocating快
- 记住preallocating快
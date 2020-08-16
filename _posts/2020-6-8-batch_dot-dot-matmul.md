---
layout:     post
title:      batch_dot-dot-matmul
subtitle:   
date:       2020-6-8
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 编程语言
---

最近在学习Transfomer模型，看了无数人的keras实现，半懵半懂，先学习下dot/batch_dot/matmul的区别吧

### dot
- dot会把x的最后一维和w的最后两维来做点积
```python
x = tf.Variable(np.random.randint(10,size=(100,200,300,20)),dtype=tf.float32)
w = tf.Variable(np.random.randint(10,size=(5,20,18)),dtype=tf.float32)
uit = K.dot(x,w)
uit.shape###[100, 200, 300, 5, 18]
```

### batch_dot
- Batchwise dot product.
```python
x = tf.Variable(np.random.randint(10,size=(100,200,300,20)),dtype=tf.float32)
w = tf.Variable(np.random.randint(10,size=(100,20,18)),dtype=tf.float32)
uit = K.batch_dot(x,w)
uit.shape###TensorShape([100, 200, 300, 18])
```
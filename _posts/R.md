---
layout:     post
title:      R
subtitle:   R
date:       2018-10-14
author:     Chunhan Li
header-img: img/post-bg-re-vs-ng2.jpg
catalog: false
tags:
    - R
---
[toc]

## 前言
这个文档用于记录学习中遇到的一些R的函数，还有一些好用的用法~（经常偶尔遇到一些特别好用的函数，过几天就忘记怎么用了）。主要用于自己的学习，所以简单的函数就稍微注释一下，不仔细写具体作用了。

## 正文
### 其他
1. split
```
split(data,data$year)##返回list
```

2. startswith [str 处理]
3. aggregate 有点像bygroup
```
aggregate(tuition.in_state~ownership,college,mean)
```

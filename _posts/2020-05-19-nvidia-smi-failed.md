---
layout:     post
title:      nvidia-smi-failed
subtitle:   
date:       2020-05-19
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 杂谈
---

[toc]

昨天用公司服务器的时候不知道为什么, nvidia-smi就报错
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```
参考[这个网站](https://blog.csdn.net/Felaim/article/details/100516282)的第二种方法可以解决

但是前提是devel版本要和内核版本一直,具体怎么做,参考[这个救星](https://www.cnblogs.com/harrymore/p/10307769.html). 然后就OK了 

整了我总共大概5小时，我真是吐了，还好解决了。
---
layout:     post
title:      Python连Oracle
subtitle:   
date:       2021-3-9
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 机器学习
---

### 前言

最后一个项目上遇到这个问题，在服务器中启动notebook后再去连接远程Oracle服务器，花了一天时间终于把这个问题搞定了，记录一下


### 过程

1. 首先要配置好jupyter notebook,这里要注意选择的端口与外部的windows要通
2. python连接oracle一定需要cx_oracle 这里需要额外安装cx_oracle/而且好像也必须要oracle 在服务器上装上
3. 刚开始遇到python 直接进入Python2了 直接在启动的jupyter notebook里面运行 cx_oracle的运行也不好使
```python
###解压cx_Oracle-6.0b2.tar.gz 进入文件夹
python setup.py install
```
4. 后来发现 source anaconda3/bin/activate 就能进入conda环境
5. Import 报错 cx_oracle DPI-1047

```
### 参考网站https://blog.csdn.net/am540/article/details/109468414
### 参考 另外的 https://blog.csdn.net/tandelin/article/details/98942995
### 修改环境变量 因为python无法定位Oracle文件
###  libclntsh.so.11.1 这个的文件夹
### export PATH=/xxxx:$LD_LIBRARY_PATH 这部不需要
```

6. import正常后发现执行语句一直卡住了
```Python
import cx_Oracle
### plsql连oracle的时候不知道这个信息 那就找oracle目录下一个叫tnsnames的文件夹
dns_tns = cx_Oracle.makedsn('10.170.36.214','1521',service_name = 'srcbfin')
con = cx_Oracle.connect('amlpoc2','amlpoc2',dns_tns)
cur = con.cursor()
sql = 'select * from a'
cur.execute(sql)###卡住了 表很小
```

7. 中间尝试了用Oracle直连
```
su -oracle ###切换oracle
echo $ORACLE_SID ###打印本地数据库SID

####连接远程oracle
su -oracle
sqlplus amlpoc2/amlpoc2@1.1.1.1:1521/srcbfin###这里是数据库名字
SELECT instance_name FROM v$instance ###查看SID
```

8. 借助sqlalchemy库连

```python
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine("oracle+cx_oracle://amlpoc2:amlpoc2@10.170.36.211:1521/srcbfin1")###这里srcbfin1是SID
sql = 'SELECT * FROM query.aml_poc_table'
df = pd.read_sql(sql,con = engine)
```


### 大功告成！！！
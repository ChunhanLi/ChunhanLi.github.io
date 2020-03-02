---
layout:     post
title:      csv跑sql in python
subtitle:   csv跑sql in python
date:       2020-03-02
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 编程语言
---



### csv跑sql in python example
```python
import sqlite3
conn = sqlite3.connect("database2.db")###任意取名
orders.to_sql('orders',conn,index=False,if_exists='replace')
sql_string = "select TRANS_DATETIME,substr(TRANS_DATETIME,12,2) from orders \
where substr(TRANS_DATETIME,12,2) in ('24') \
limit 10"
test = pd.read_sql(sql_string,conn)
test
```
---
layout:     post
title:      Python写xlsx
subtitle:   Python写xlsx
date:       2020-03-02
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 编程语言
---

### 利用Python生成xlsx example

```python
writer = pd.ExcelWriter('orders_rule1-10.xlsx')
temp1 = pd.DataFrame({'A':[1,2,3],'B':[4,5,6]})
temp1.to_excel(writer,index=False,sheet_name='csv1')
temp1.to_excel(writer,index=False,sheet_name='csv2')
writer.save()
```


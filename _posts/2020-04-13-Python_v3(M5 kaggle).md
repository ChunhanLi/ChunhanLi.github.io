---
layout:     post
title:      Python_v3(M5 kaggle)
subtitle:   Python
date:       2020-04-13
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 编程语言
---

[toc]

#### tqdm在notebook里打断后出现的错误
- https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook-prints-new-progress-bars-repeatedly/60067890#60067890
- https://github.com/jupyter/notebook/issues/2214
- from tqdm.notebook import tqdm可以解决
- pip install --upgrade tqdm

#### lgb 直接plot_importance
- lgb.plot_importance(lgb_re1, importance_type="split", precision=0, height=0.5, figsize=(6, 10))
- lgb.plot_importance(lgb_re1, importance_type="gain", precision=0, height=0.5, figsize=(6, 10));
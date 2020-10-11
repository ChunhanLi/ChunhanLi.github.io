### pytorch

记录一下入坑pytorch的一些小疑惑

- BCEWithLogitsLoss和BCELoss的区别(BCE是要在模型里sigmoid然后直接logloss/BCEwith会自动先sigmoid然后logloss)
- BCELOSS不准 别用！！！！
- model.train和Model.eval模式会影响BN和Dropout
- optimizer.zero_grad()梯度初始化为0
- loss.backward()反向传播
- optimizer.step()模型参数更新？
- pytorch train loss每次显示的是每个batch的loss
- Dataloader 另外的的用法TensorDataset(https://www.kaggle.com/hengzheng/pytorch-starter/comments)

#### 复现代码
```python
#### 每次运行前都得运行
#### 不是运行一次固定seed 就能连续跑两次
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=47)
```

#### 学习率调整
- torch.optim.lr_scheduler.ReduceLROnPlateau(https://blog.csdn.net/weixin_40100431/article/details/84311430)
- 要在step()里面加需要优化的val_loss
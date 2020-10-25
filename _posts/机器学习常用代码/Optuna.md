- 网站 https://optuna.readthedocs.io/en/stable/tutorial/001_first.html#first
```python
import optuna
def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2
study = optuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_params)
study.best_value

def objective(trial):
    # Categorical parameter
    optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])

    # Int parameter
    num_layers = trial.suggest_int('num_layers', 1, 3)

    # Uniform parameter
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

    # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Discrete-uniform parameter
    drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)

    ...
```

```python
### my demo
def objective(trial):
    params={
        "hidden_size1": trial.suggest_int("hidden_size",256,2028),
        "dropout1": trial.suggest_uniform("dropout",0.3,0.7),
        "hidden_size2": trial.suggest_int("hidden_size",256,2028),
        "dropout2": trial.suggest_uniform("dropout",0.3,0.7),
        "dropout3": trial.suggest_uniform("dropout",0.2,0.8),
    }
    

    class MoaModel(nn.Module):
        def __init__(self,num_columns):
            super(MoaModel,self).__init__()
            self.dense1 = nn.utils.weight_norm(nn.Linear(num_columns,params['hidden_size1']))
            self.dropout1 = nn.Dropout(params['dropout1'])
            self.batch_norm1 = nn.BatchNorm1d(num_columns)
            self.prelu1 = nn.PReLU()

            self.dense2 = nn.utils.weight_norm(nn.Linear(params['hidden_size1'],params['hidden_size2']))
            self.dropout2 = nn.Dropout(params['dropout2'])
            self.batch_norm2 = nn.BatchNorm1d(params['hidden_size1'])
            self.prelu2 = nn.PReLU()

            self.dense3 = nn.utils.weight_norm(nn.Linear(params['hidden_size2'],206))
            self.dropout3 = nn.Dropout(params['dropout3'])
            self.batch_norm3 = nn.BatchNorm1d(params['hidden_size2'])

        def forward(self,x):
            x = self.batch_norm1(x)
            x = self.dropout1(x)
            x = self.prelu1(self.dense1(x))

            x = self.batch_norm2(x)
            x = self.dropout2(x)
            x = self.prelu2(self.dense2(x))

            x = self.batch_norm3(x)
            x = self.dropout3(x)
            x = self.dense3(x)

            return x

    overall_mean = []
    overall_train_oof = np.zeros(y_train.shape)
    seed_list = [42,47,1103]
    for seed in seed_list:
        #print('seeds',seed,'begins**************************************************')
        seed_everything(seed=seed)

        kf = MultilabelStratifiedKFold(n_splits=5,random_state=seed,shuffle=True)


        batch_size = 128
        batch_size_val = 1024*4
        train_epochs = 100
        Early_stop = True
        Early_stop_step = 5

        k = 1
        train_oof = np.zeros(y_train.shape)
        test_preds_fold = np.zeros((X_test.shape[0],206))
        best_loss_list = []
        for train_index,test_index in kf.split(X_train,y_train):
            k+=1
            X_train2, y_train2 = X_train.iloc[train_index,:],y_train.iloc[train_index,:]
            X_test2, y_test2 = X_train.iloc[test_index,:],y_train.iloc[test_index,:]

            train_set = MoaDataset(X_train2,y_train2)
            val_set = MoaDataset(X_test2,y_test2)
            dataloaders = {
                'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
                'val': DataLoader(val_set, batch_size=batch_size_val, shuffle=False)
            }
            model = MoaModel(X_train2.shape[1]).to(device)
            optimizer = optim.Adam(model.parameters(),weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=False)
            loss_func = nn.BCEWithLogitsLoss()
            early_step = 0
            best_loss = np.inf
            for epoch in range(train_epochs):
                model.train()
                start_time = time.time()
                for x,y in dataloaders['train']:
                    optimizer.zero_grad()
                    x = x.to(device)
                    y = y.to(device)
                    preds = model(x)
                    loss = loss_func(preds,y)
                    label_smoothing = 0.001
                    y_smo = y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
                    loss  = F.binary_cross_entropy_with_logits(preds, y_smo.type_as(preds))               
                    #print(preds.size())
                    #print(y.size())
                    ##optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #print(loss.item())

                model.eval()
                val_loss = 0
                val_kfold = np.zeros(y_test2.shape)
                for i, (x_val, y_val) in enumerate(dataloaders['val']):
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    y_val_pred = model(x_val).detach()
                    loss_func_sum = nn.BCEWithLogitsLoss(reduction = 'sum')
                    #print(loss_func_sum(y_val_pred, y_val).item())
                    val_loss += loss_func_sum(y_val_pred, y_val).item() / (y_test2.shape[0]*y_test2.shape[1])
                    val_kfold[i * batch_size_val:(i+1) * batch_size_val,:] = y_val_pred.cpu().numpy()
                elapsed_time = time.time() - start_time
                scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    train_oof[test_index,:] = val_kfold
                    torch.save(model.state_dict(), f"FOLD{k-1}_seed{seed}.pkl")
                    early_step = 0
                elif Early_stop:
                    early_step += 1
                    if (early_step >= Early_stop_step):
                        best_loss_list.append(best_loss)
                        break
        overall_mean.append(round(np.mean(best_loss_list),10))
        overall_train_oof += torch.FloatTensor(train_oof).numpy()/(len(seed_list))
    return np.mean(overall_mean)

```
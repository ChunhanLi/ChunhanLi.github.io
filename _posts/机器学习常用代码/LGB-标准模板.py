skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=47)
fea_impor = 0
oof_train = np.zeros(X_train.shape[0])
y_pred = np.zeros(X_test.shape[0])
f1_fold_list = []
k = 0
for train_index,test_index in skf.split(X_train,y_train):
    k+=1
    print(f'{k}folds begins******************************')
    X_train2 = X_train.iloc[train_index,:]
    y_train2 = y_train.iloc[train_index]
    X_test2 = X_train.iloc[test_index,:]
    y_test2 = y_train.iloc[test_index]
    clf = lgb.LGBMClassifier(n_estimators=10000, random_state=47,learning_rate=0.01,importance_type = 'gain',
                 n_jobs = -1,metric = 'None')
    # clf = lgb.LGBMClassifier(n_estimators=10000, random_state=47+_,learning_rate=0.01,importance_type = 'gain',
    #                     n_jobs = -1,num_leaves=20,bagging_freq=1,colsample_bytree=0.5,subsample=1,min_child_weight = 0.1,
    #                             min_child_samples = 250,reg_alpha = 1.5,reg_lambda = 1,metric = 'None',min_split_gain = 0)
    clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)],\
            eval_metric=lambda y_true,y_pred:f1_score_custom(y_true,y_pred),early_stopping_rounds=100,verbose=50)
    tmp = clf.predict_proba(X_test2)[:,1]
    oof_train[test_index] = tmp
    f1_loss = f1_score(y_test2,tmp.round())
    f1_fold_list.append(f1_loss)
    y_pred += clf.predict_proba(X_test)[:,1]/skf.n_splits
    fea_impor += clf.feature_importances_/skf.n_splits
for _ in f1_fold_list:
    print(_)
print('mean f1',np.mean(f1_fold_list))
print('oof f1',f1_score(y_train,oof_train.round()))
#### demo要改
def get_feature_importance(X_train_raw,X_test_raw,prefix,shuffle=False):
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    if shuffle:
        X_test = np.random.permutation(X_test)
    params = {'bagging_freq':1,
    'num_leaves': 370,
     'subsample': 0.9967584074071576,
     'colsample_bytree': 0.9134179494490106,
     'min_child_weight': 0.5400296245490022,
     'min_child_samples': 30,
     'reg_alpha': 2.7153164704173993,
     'reg_lambda': 1.2598378046329661,
     'min_split_gain': 0.15398430096611754}
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=47)
    fea_impor = 0
    oof_train = np.zeros((X_train.shape[0],3))
    y_pred = np.zeros((X_test.shape[0],3))
    kappa_fold_list = []
    k = 0
    for train_index,test_index in skf.split(X_train,y_train):
        k+=1
        #print(f'{k}folds begins******************************')
        X_train2 = X_train.iloc[train_index,:]
        y_train2 = y_train.iloc[train_index]
        X_test2 = X_train.iloc[test_index,:]
        y_test2 = y_train.iloc[test_index]
        clf = lgb.LGBMClassifier(n_estimators=10000, random_state=47,learning_rate=0.01,importance_type = 'gain',metric='None',
                     n_jobs = -1,**params)

        clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)],early_stopping_rounds=100,verbose=0,\
                eval_metric=kappa_custom)
        tmp = clf.predict_proba(X_test2)
        oof_train[test_index,:] = tmp
        kappa_loss = cohen_kappa_score(y_test2,tmp.argmax(axis=1))
        #print(f'fold{k} kappa_loss',kappa_loss)
        kappa_fold_list.append(kappa_loss)
        y_pred += clf.predict_proba(X_test)/skf.n_splits
        fea_impor += clf.feature_importances_/skf.n_splits
    return pd.DataFrame(fea_impor,columns=[prefix+'_importance'],index=X_train.columns)

nums = 10
true_impor = get_feature_importance(X_train,X_test,'true',shuffle=False)
null_impor = pd.DataFrame(columns=[prefix+'_importance'],index=X_train.columns)
for num in range(nums):
    print(num)
    tmp = get_feature_importance(X_train,X_test,str(num),shuffle=True)
    true_impor = true_impor.merge(tmp,left_index=True,right_index=True)

###用真实/75%分位数去评估
gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))
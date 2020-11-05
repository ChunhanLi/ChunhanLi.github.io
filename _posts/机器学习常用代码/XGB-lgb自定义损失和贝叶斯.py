#### lgb自定义损失
def f1_score_custom(y_true,y_pred):
    #print(y_pred,y_true.shape)
    y_pred = y_pred.round()
    return 'f1', f1_score(y_true,y_pred), True
#### 多分类 需要自己reshape
def kappa_custom(y_true,y_pred):
    y_pred = np.reshape(y_pred,(-1,3),'F')
    y_pred = y_pred.argmax(axis=1)
    return 'kappa', cohen_kappa_score(y_true,y_pred), True
#### xgb自定义损失 注意f1_score的负号; 
clf = xgb.XGBClassifier(n_estimators=50, random_state=47,learning_rate=0.06,\
                            importance_type = 'gain',n_jobs = -1,metric='None')
def f1_score_custom(preds, dtrain):
    labels = dtrain.get_label()
    y_pred = preds.round()
    return 'f1_macro', -f1_score(labels,y_pred,average='macro')
clf.fit(X_train2,y_train2,eval_set = [(X_test2,y_test2)],
        early_stopping_rounds=200,verbose=50,eval_metric=f1_score_custom)


#### 贝叶斯
def xgb_eval(subsample1,colsample_bytree1,max_depth1,min_child_weight1,reg_alpha1,reg_lambda1,gamma1,colsample_bylevel1):
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=47)
    fea_impor = 0
    k = 1
    oof = np.zeros(y_train.shape)
    pred = 0
    pred_os = 0
    score_list = []
    for train_index,test_index in kf.split(X_train,y_train):
        #print(f'fold_{k}*********************************************')
        k+=1
        X_train2 = X_train.iloc[train_index,:]
        y_train2 = y_train.iloc[train_index]
        X_test2 = X_train.iloc[test_index,:]
        y_test2 = y_train.iloc[test_index]
        clf = xgb.XGBClassifier(n_estimators=10000, random_state=47,learning_rate=0.2,\
                                 importance_type = 'total_gain',n_jobs = -1,metric='None',\
                               max_depth=int(max_depth1),colsample_bytree=colsample_bytree1
                                 ,subsample=subsample1,min_child_weight = min_child_weight1,
                                 gamma = int(gamma1),reg_alpha = reg_alpha1,reg_lambda = reg_lambda1,\
                               colsample_bylevel = colsample_bylevel1)
        def f1_score_custom(preds, dtrain):
            labels = dtrain.get_label()
            y_pred = preds.round()
            return 'f1_macro', -f1_score(labels,y_pred,average='macro')
        clf.fit(X_train2,y_train2,eval_set = [(X_test2,y_test2)],
                early_stopping_rounds=200,verbose=0,eval_metric=f1_score_custom)
        temp = clf.predict_proba(X_test2)[:,1]
        oof[test_index] = temp
        score_list.append(f1_score(y_test2,temp.round(),average='macro'))
        fea_impor += clf.feature_importances_/kf.n_splits
        pred += clf.predict_proba(X_test_chusai)/kf.n_splits
        #pred_os +=clf.predict_proba(X_test_chusai_sample)/kf.n_splits
        #break
    #print('Mean f1_macro:',np.mean(score_list))
    #print('oof',f1_score(y_train,oof.round(),average='macro'))
    return f1_score(y_test_chusai,pred[:,1].round(),average='macro')

xgb_opt = BayesianOptimization(xgb_eval, {
                                          'max_depth1': (2,11),
                                          'subsample1': (0.05, 1),
                                          'colsample_bytree1': (0.05,1),
                                            'min_child_weight1':(0,5),
                                            'reg_alpha1':(0,5),
                                            'gamma1':(0,5),
                                        'reg_lambda1':(0,5),
                                        'colsample_bylevel1':(0.05, 1)
                                        })
xgb_opt.maximize(n_iter=30, init_points=5)

#lgb_opt.max['target']
xgb_opt.max['params']
xgb_opt.max['target']
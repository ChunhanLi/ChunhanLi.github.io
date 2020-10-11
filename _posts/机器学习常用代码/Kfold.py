kf = KFold(n_splits=5,shuffle=True,random_state=47)
X_train = user.iloc[:,3:]
y_train_age = user['age']
y_train_gender = user['gender']

fea_impor = 0
k = 1
pred_gender = 0
y_train_pred = np.zeros(y_train_age.shape)
score_list = []
for train_index,test_index in kf.split(X_train,y_train_age):
    print(f'fold_{k}*********************************************')
    k+=1
    X_train2 = X_train.iloc[train_index,:]
    y_train2 = y_train_age.iloc[train_index]
    X_test2 = X_train.iloc[test_index,:]
    y_test2 = y_train_age.iloc[test_index]
    clf = lgb.LGBMClassifier(n_estimators=10000, random_state=47,learning_rate=0.1,importance_type = 'gain',n_jobs = -1,metric='None',\
                           subsample = 0.9559480594711868,reg_lambda=1.2446652415322004,reg_alpha=0.6535477765318327,\
                           num_leaves=18,min_split_gain= 0.23165748340318615,min_child_weight=0.9074928187791584,\
                            min_child_samples = 330,colsample_bytree =  0.665959605144615)
    def acc_score(y_true,y_pred):
        y_pred = y_pred.reshape(-1,10,order = 'F').argmax(axis=1)
        return 'Accuracy', accuracy_score(y_true,y_pred), True
    clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)],early_stopping_rounds=100,verbose=50\
            ,eval_metric=lambda y_true,y_pred:acc_score(y_true,y_pred))
    temp = clf.predict(X_test2)
    y_train_pred[test_index] = temp
    score_list.append(accuracy_score(y_test2,temp))
    fea_impor += clf.feature_importances_/kf.n_splits
    pred_gender += clf.predict_proba(X_test)/kf.n_splits
    break
print('Mean Accuracy:',np.mean(score_list))
print('OOF Accuracy:',accuracy_score(y_train_age,y_train_pred))



kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=47)
X_train = train_user.drop(['phone_no_m','label'],axis=1)
y_train =  train_user['label']
X_test = test_user.drop(['phone_no_m'],axis=1)
#y_test =  = test_user['label']


fea_impor = 0
k = 1
oof = np.zeros(y_train.shape)
pred = 0
score_list = []
for train_index,test_index in kf.split(X_train,y_train):
    print(f'fold_{k}*********************************************')
    k+=1
    X_train2 = X_train.iloc[train_index,:]
    y_train2 = y_train.iloc[train_index]
    X_test2 = X_train.iloc[test_index,:]
    y_test2 = y_train.iloc[test_index]
    clf = lgb.LGBMClassifier(n_estimators=10000, random_state=47,learning_rate=0.05,importance_type = 'gain',n_jobs = -1,metric='None')
    def f1_score_custom(y_true,y_pred):
        #print(y_pred,y_true.shape)
        y_pred = y_pred.round()
        return 'f1_macro', f1_score(y_true,y_pred,average='macro'), True
    clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)],early_stopping_rounds=100,verbose=50,eval_metric=lambda y_true,y_pred:f1_score_custom(y_true,y_pred))
    temp = clf.predict_proba(X_test2)[:,1]
    oof[test_index] = temp
    score_list.append(f1_score(y_test2,temp.round(),average='macro'))
    fea_impor += clf.feature_importances_/kf.n_splits
    pred += clf.predict_proba(X_test)/kf.n_splits
    #break
print('Mean f1_macro:',np.mean(score_list))
print('OOF f1_macro:',f1_score(y_train,oof.round(),average='macro'))
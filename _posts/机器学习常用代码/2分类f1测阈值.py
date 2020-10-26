threshold_list = []
f1_score_list = []
score_df = pd.DataFrame()
for _ in tqdm(np.arange(0.05,1,0.01)):
    threshold_list.append(_)
    f1_score_list.append(f1_score(y_train,pd.Series(oof_train).map(lambda x:1 if x>=_ else 0)))
score_df['thres'] = threshold_list
score_df['f1_score'] = f1_score_list
score_df.sort_values('f1_score',ascending=False)
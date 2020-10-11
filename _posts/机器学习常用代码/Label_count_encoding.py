#### labelEncoder
for feature in cat_list:
    label_encod = LabelEncoder()
    label_encod.fit(list(X_train[feature].astype(str).values) + list(X_test[feature].astype(str).values))
    X_train[feature] = label_encod.transform(list(X_train[feature].astype(str).values))
    X_test[feature] = label_encod.transform(list(X_test[feature].astype(str).values))

#### count encoding

for i in count_columns:
    train[i+'_count_full'] = train[i].map(pd.concat([train[i], test[i]], ignore_index=True).value_counts(dropna=False))
    test[i+'_count_full'] = test[i].map(pd.concat([train[i], test[i]], ignore_index=True).value_counts(dropna=False))

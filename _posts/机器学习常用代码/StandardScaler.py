from sklearn.preprocessing import StandardScaler
sds = StandardScaler()
X_train = pd.DataFrame(sds.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(sds.transform(X_test),columns=X_test.columns)
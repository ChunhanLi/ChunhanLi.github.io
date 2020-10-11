#### https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82055
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=None,max_df = 0.9,min_df = 0.1)
df_bag = pd.DataFrame(df[[groupby, target]])
df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))})
df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))
tfidf_full_vector = tfidf_vec.fit_transform(df_bag[target + '_list'])
svd_vec = TruncatedSVD(n_components=64, algorithm='arpack')
svd_vec.fit(tfidf_full_vector)
te = svd_vec.transform(tfidf_full_vector)

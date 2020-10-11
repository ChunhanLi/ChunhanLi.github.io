tokenizer = Tokenizer(lower=False, char_level=False, split=',')
###把所有词字典吃点进去
tokenizer.fit_on_texts(embed_df[target])
### test to sequence
X_train_seq = tokenizer.texts_to_sequences(df_bag[:900000]['list'])
X_test_seq = tokenizer.texts_to_sequences(df_bag[900000:]['list'])
### padding
X_train = pad_sequences(X_train_seq, maxlen=maxlen, value=0)
X_test = pad_sequences(X_test_seq, maxlen=maxlen, value=0)
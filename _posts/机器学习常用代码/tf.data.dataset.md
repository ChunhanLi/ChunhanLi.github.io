```python
####处理变长list输入tf.data.dataset
#### https://blog.csdn.net/tianzhiya121/article/details/89206421
#### 读入数据
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
####使用长度等于数据集大小的buffer size，打乱数据集。这确保了良好的改组。
dataset = dataset.shuffle(len(filenames))
#### 数据增强
dataset = dataset.map(parse_function, num_parallel_calls=4)
#### 根据可用的CPU动态设置并行调用的数量
dataset = dataset.map(train_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#### 抽取batch_szie
dataset = dataset.batch(batch_size)
### GPU执行在当前批次执行前向或者后向传播时，我们希望CPU处理下一个批次的数据，以便于数据批次能够迅速被GPU使用。
### 我们希望GPU被完全、时刻用于训练。我们称这种机制为消费者/生产者重叠，消费者是GPU，生产者是CPU。
### 使用tf.data，你可以轻易的做到只一点，只需要在通道末尾调用dataset.prefetch(1)。这将总是预取一个批次的数据，并且保证总有一个数据准备好被消耗。
dataset = dataset.prefetch(1)
```
### 记录一次失败的tf.data.dataset使用

```python
def model_lstm(embed_cre,embed_ad,embed_adv,embed_prod):

    K.clear_session()
    # The embedding layer containing the word vectors
    emb_layer_cr = Embedding(
        input_dim=embed_cre.shape[0],
        output_dim=embed_cre.shape[1],
        weights=[embed_cre],
        input_length=100,
        trainable=False
    )
    emb_layer_ad = Embedding(
        input_dim=embed_ad.shape[0],
        output_dim=embed_ad.shape[1],
        weights=[embed_ad],
        input_length=100,
        trainable=False
    )
    emb_layer_adv = Embedding(
        input_dim=embed_adv.shape[0],
        output_dim=embed_adv.shape[1],
        weights=[embed_adv],
        input_length=100,
        trainable=False
    )
    emb_layer_pro = Embedding(
        input_dim=embed_prod.shape[0],
        output_dim=embed_prod.shape[1],
        weights=[embed_prod],
        input_length=100,
        trainable=False
    )
    #     emb_layer_ind = Embedding(
    #         input_dim=embed_industry.shape[0],
    #         output_dim=embed_industry.shape[1],
    #         weights=[embed_industry],
    #         input_length=20,
    #         trainable=False
    #     )

    
    
    lstm_layer_cr = Bidirectional(
            LSTM(128, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))
    lstm_layer_ad = Bidirectional(
            LSTM(128, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))
    lstm_layer_adv = Bidirectional(
            LSTM(128, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))
    lstm_layer_prod = Bidirectional(
            LSTM(64, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))
#     lstm_layer_ind = Bidirectional(
#             LSTM(16, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))
    # 1D convolutions that can iterate over the word vectors
    conv1_cr = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_ad = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_adv = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_prod = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu',)

    seq_cr = Input(shape=(100,))
    seq_ad = Input(shape=(100,))
    seq_adv = Input(shape=(100,))
    seq_prod = Input(shape=(100,))

    #seq_ind = Input(shape=(20,))


    emb_cr = emb_layer_cr(seq_cr)
    emb_ad = emb_layer_ad(seq_ad)
    emb_adv = emb_layer_adv(seq_adv)
    emb_prod = emb_layer_pro(seq_prod)
    #emb_ind = emb_layer_ind(seq_ind)
    
    lstm_cr = lstm_layer_cr(emb_cr)
    lstm_ad = lstm_layer_ad(emb_ad)
    lstm_adv = lstm_layer_adv(emb_adv)
    lstm_prod = lstm_layer_prod(emb_prod)

    
    
    conv1a_cr = conv1_cr(lstm_cr)
    conv1a_ad = conv1_ad(lstm_ad)
    conv1a_adv = conv1_adv(lstm_adv)
    conv1a_prod = conv1_prod(lstm_prod)
    
    # Run through CONV + GAP layers
    gap1a_cr = GlobalAveragePooling1D()(conv1a_cr)
    gmp1a_cr = GlobalMaxPool1D()(conv1a_cr)

    gap1a_ad = GlobalAveragePooling1D()(conv1a_ad)
    gmp1a_ad = GlobalMaxPool1D()(conv1a_ad)
    
    gap1a_adv = GlobalAveragePooling1D()(conv1a_adv)
    gmp1a_adv = GlobalMaxPool1D()(conv1a_adv)
    
    gap1a_prod = GlobalAveragePooling1D()(conv1a_prod)
    gmp1a_prod = GlobalMaxPool1D()(conv1a_prod)
    
    gap1a_cr_lstm = GlobalAveragePooling1D()(lstm_cr)
    gmp1a_cr_lstm = GlobalMaxPool1D()(lstm_cr)

    gap1a_ad_lstm = GlobalAveragePooling1D()(lstm_ad)
    gmp1a_ad_lstm = GlobalMaxPool1D()(lstm_ad)
    
    gap1a_adv_lstm = GlobalAveragePooling1D()(lstm_adv)
    gmp1a_adv_lstm = GlobalMaxPool1D()(lstm_adv)
    
    gap1a_prod_lstm = GlobalAveragePooling1D()(lstm_prod)
    gmp1a_prod_lstm = GlobalMaxPool1D()(lstm_prod)
    
    #gap1a_ind = GlobalAveragePooling1D()(conv1a_ind)
    #gmp1a_ind = GlobalMaxPool1D()(conv1a_ind)
    
    merge1 = concatenate([gap1a_cr, gmp1a_cr,gap1a_ad,gmp1a_ad,\
                         gap1a_adv,gmp1a_adv,gap1a_prod,gmp1a_prod,\
                         gap1a_cr_lstm,gmp1a_cr_lstm,gap1a_ad_lstm,gmp1a_ad_lstm,\
                         gap1a_adv_lstm,gmp1a_adv_lstm,gap1a_prod_lstm,gmp1a_prod_lstm])

    
    x = Dropout(0.3)(merge1)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu',)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu',)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu',)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    pred = Dense(10, activation='softmax')(x)
    model = models.Model(inputs=[seq_cr,seq_ad,seq_adv,seq_prod], outputs=pred)
#     model_gpu2=multi_gpu_model(model, gpus=2)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def shuffle_op(seq1,seq2,seq3,seq4, label):
    seq1 = tf.random.shuffle(seq1)
    seq2 = tf.random.shuffle(seq2)
    seq3 = tf.random.shuffle(seq3)
    seq4 = tf.random.shuffle(seq4)
    return seq1,seq2,seq3,seq4, label
def pad_zeros(a,max_len):
    if tf.shape(a)<max_len:
        zero_padding = tf.zeros(max_len - tf.shape(a), dtype=a.dtype)
        return tf.concat([zero_padding,a],0)
    else:
        return a[:max_len]
def pad_op(seq1,seq2,seq3,seq4,label,max_len1,max_len2,max_len3,max_len4):
    return tuple([pad_zeros(seq1,max_len1),pad_zeros(seq2,max_len2),
                  pad_zeros(seq3,max_len3),pad_zeros(seq4,max_len4)]),label


def dataset_train(df_bag_train,label_train,max_len1,max_len2,max_len3,max_len4,batch_size):
    sent1 = df_bag_train.list_creative_id.values
    sent2 = df_bag_train.list_ad_id.values
    sent3 = df_bag_train.list_advertiser_id.values
    sent4 = df_bag_train.list_product_id.values

    labels = label_train

    def generator():
        for s1, s2,s3,s4 ,l in zip(sent1, sent2,sent3,sent4, labels):
            yield s1,s2,s3,s4,l

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.int32,tf.int32,tf.int32,tf.int32, tf.int16))
    dataset = dataset.shuffle(df_bag_train.shape[0])
    dataset = dataset.map(shuffle_op, num_parallel_calls=4)
    dataset = dataset.map(lambda s1,s2,s3,s4,y:pad_op(s1,s2,s3,s4,y,max_len1,max_len2,max_len3,max_len4), num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset
def dataset_valid(df_bag_train,label_train,max_len1,max_len2,max_len3,max_len4,batch_size):
    sent1 = df_bag_train.list_creative_id.values
    sent2 = df_bag_train.list_ad_id.values
    sent3 = df_bag_train.list_advertiser_id.values
    sent4 = df_bag_train.list_product_id.values

    labels = label_train

    def generator():
        for s1, s2,s3,s4 ,l in zip(sent1, sent2,sent3,sent4, labels):
            yield s1,s2,s3,s4,l

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.int32,tf.int32,tf.int32,tf.int32, tf.int16))
    #dataset = dataset.shuffle(df_bag_train.shape[0])
    dataset = dataset.map(shuffle_op, num_parallel_calls=8)
    dataset = dataset.map(lambda s1,s2,s3,s4,y:pad_op(s1,s2,s3,s4,y,max_len1,max_len2,max_len3,max_len4), num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=47)
user =  pd.read_csv('raw_data_chusai/user.csv')
#X_train = user.iloc[:,3:]
#print(X_train.shape)
y_train_age = user['age'] - 1
y_train_gender = user['gender'] - 1
from keras.utils.np_utils import to_categorical
y_train_age = to_categorical(y_train_age, num_classes=10)
fea_impor = 0
k = 1
pred_gender = 0
y_train_pred = np.zeros(y_train_age.shape)
y_train_pred_prob = np.zeros((y_train_age.shape[0],10))
#y_train_pred_prob = np.zeros((y_train_gender.shape[0],1))
score_list = []
for train_index,test_index in kf.split(df_bag[:900000],user['age'] - 1):
    print(f'fold_{k}*********************************************')
    file_name = f'model_atten_v30_fold{k}.hdf5'
    k+=1
    X_train2 = df_bag[:900000].iloc[train_index,:]
    y_train2 = y_train_age[train_index]

    X_train2 = df_bag[:900000].iloc[train_index,:]
    y_train2 = y_train_age[train_index]
    
    #X_test2 = df_bag.iloc[test_index,:]
    X_test2 = df_bag[:900000].iloc[test_index,:]
    y_test2 = y_train_age[test_index]
    

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=3, min_lr=0.0004, verbose=1)
    model_checkpoint = ModelCheckpoint(file_name,save_best_only=True, verbose=1, monitor='val_acc', mode='auto')
    es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, verbose=10,restore_best_weights=True)
    #reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=2, min_lr=0.0005, verbose=1)

    train_generator = dataset_train(X_train2,y_train2,100,100,100,100,batch_size=1024)
    valid_generator = dataset_valid(X_test2,y_test2,100,100,100,100,batch_size=4096)

    model = model_lstm(embed_cre,embed_ad,embed_adv,embed_prod)
    model.fit_generator(train_generator,\
              validation_data  = valid_generator, 
               epochs=200,verbose=1, callbacks=[es,reduce_lr],steps_per_epoch = X_train2.shape[0]//1024)
    break
    #y_train_pred_prob[test_index,:] = model.predict([X_test_cre2,X_test_ad2,X_test_adv2,X_test_prod2])
    #pred_gender+= model.predict([X_test_cre,X_test_ad,X_test_adv,X_test_prod])/kf.n_splits
```
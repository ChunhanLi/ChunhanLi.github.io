####据说是更快的LSTM Only work in GPU
#### 不过CuDNNLSTM 没有之前LSTM的两个dropout
#### tensorflow.keras下运行可以
#### fit_generator 在tf.keras.utils.Sequence不能和multiprocessing连用?
#### -------> https://github.com/stellargraph/stellargraph/issues/1206
#### -------> https://github.com/stellargraph/stellargraph/issues/1006
#### -------> 解决办法用tf.data.Dataset代替generator 但是我看了半天不会这个...
#### 测试结果
#### -------> CuDNNLSTM 用fit_generator w/o multiprocessing 速度和LSTM w multiprocessing差不多 predict好像变快很多
#### -------> CuDNNLSTM 普通fit 比 LSTM 快3倍 predict没测
import tensorflow as tf

inp = tf.keras.layers.Input(shape=(10,))
emb = tf.keras.layers.Embedding(20, 4)(inp)
x2 = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64, return_sequences=True))(emb)
max_pl = tf.keras.layers.GlobalMaxPooling1D()(x2)
x = tf.keras.layers.Dense(16, activation="relu")(max_pl)
x = tf.keras.layers.Dropout(0.1)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.models.Model(inputs=inp, outputs=output)
model.summary()

####
def model_lstm(embed_cre,embed_ad,embed_adv,embed_prod,embed_industry):

    K.clear_session()
    # The embedding layer containing the word vectors
    emb_layer_cr = tf.keras.layers.Embedding(
        input_dim=embed_cre.shape[0],
        output_dim=embed_cre.shape[1],
        weights=[embed_cre],
        input_length=100,
        trainable=False
    )
    emb_layer_ad = tf.keras.layers.Embedding(
        input_dim=embed_ad.shape[0],
        output_dim=embed_ad.shape[1],
        weights=[embed_ad],
        input_length=100,
        trainable=False
    )
    emb_layer_adv = tf.keras.layers.Embedding(
        input_dim=embed_adv.shape[0],
        output_dim=embed_adv.shape[1],
        weights=[embed_adv],
        input_length=50,
        trainable=False
    )
    emb_layer_pro = tf.keras.layers.Embedding(
        input_dim=embed_prod.shape[0],
        output_dim=embed_prod.shape[1],
        weights=[embed_prod],
        input_length=50,
        trainable=False
    )
    emb_layer_ind = tf.keras.layers.Embedding(
        input_dim=embed_industry.shape[0],
        output_dim=embed_industry.shape[1],
        weights=[embed_industry],
        input_length=20,
        trainable=False
    )

    
    
    lstm_layer_cr = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(128, return_sequences=True))
    lstm_layer_ad = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(128,return_sequences=True))
    lstm_layer_adv = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64,return_sequences=True))
    lstm_layer_prod = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64,return_sequences=True))
    lstm_layer_ind = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(16,return_sequences=True))
    # 1D convolutions that can iterate over the word vectors
    conv1_cr = tf.keras.layers.Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_ad = tf.keras.layers.Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_adv = tf.keras.layers.Conv1D(filters=64, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_prod = tf.keras.layers.Conv1D(filters=64, kernel_size=1,
                   padding='same', activation='relu',)
    conv1_ind = tf.keras.layers.Conv1D(filters=16, kernel_size=1,
                   padding='same', activation='relu',)

    seq_cr = tf.keras.layers.Input(shape=(100,))
    seq_ad = tf.keras.layers.Input(shape=(100,))
    seq_adv = tf.keras.layers.Input(shape=(50,))
    seq_prod = tf.keras.layers.Input(shape=(50,))
    seq_ind = tf.keras.layers.Input(shape=(20,))


    emb_cr = emb_layer_cr(seq_cr)
    emb_ad = emb_layer_ad(seq_ad)
    emb_adv = emb_layer_adv(seq_adv)
    emb_prod = emb_layer_pro(seq_prod)
    emb_ind = emb_layer_ind(seq_ind)
    
    lstm_cr = lstm_layer_cr(emb_cr)
    lstm_ad = lstm_layer_ad(emb_ad)
    lstm_adv = lstm_layer_adv(emb_adv)
    lstm_prod = lstm_layer_prod(emb_prod)
    lstm_ind = lstm_layer_ind(emb_ind)
    
    
    
    conv1a_cr = conv1_cr(lstm_cr)
    conv1a_ad = conv1_ad(lstm_ad)
    conv1a_adv = conv1_adv(lstm_adv)
    conv1a_prod = conv1_prod(lstm_prod)
    conv1a_ind = conv1_ind(lstm_ind)
    # Run through CONV + GAP layers
    gap1a_cr = tf.keras.layers.GlobalAveragePooling1D()(conv1a_cr)
    gmp1a_cr = tf.keras.layers.GlobalMaxPool1D()(conv1a_cr)

    gap1a_ad = tf.keras.layers.GlobalAveragePooling1D()(conv1a_ad)
    gmp1a_ad = tf.keras.layers.GlobalMaxPool1D()(conv1a_ad)
    
    gap1a_adv = tf.keras.layers.GlobalAveragePooling1D()(conv1a_adv)
    gmp1a_adv = tf.keras.layers.GlobalMaxPool1D()(conv1a_adv)
    
    gap1a_prod = tf.keras.layers.GlobalAveragePooling1D()(conv1a_prod)
    gmp1a_prod = tf.keras.layers.GlobalMaxPool1D()(conv1a_prod)
    
    gap1a_ind = tf.keras.layers.GlobalAveragePooling1D()(conv1a_ind)
    gmp1a_ind = tf.keras.layers.GlobalMaxPool1D()(conv1a_ind)
    
    merge1 = tf.keras.layers.concatenate([gap1a_cr, gmp1a_cr,gap1a_ad,gmp1a_ad,\
                         gap1a_adv,gmp1a_adv,gap1a_prod,gmp1a_prod,\
                         gap1a_ind,gmp1a_ind])

    
    x = tf.keras.layers.Dropout(0.3)(merge1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu',)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu',)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu',)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    pred = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[seq_cr,seq_ad,seq_adv,seq_prod,seq_ind], outputs=pred)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
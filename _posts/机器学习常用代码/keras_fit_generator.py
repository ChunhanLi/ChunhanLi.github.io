### model.fit
### -----> 整个训练集可以放入RAM
### model.fit_generator
### -----> 真实世界的数据集通常太大而无法放入内存中
### -----> 它们也往往具有挑战性，要求我们执行数据增强以避免过拟合并增加我们的模型的泛化能力
### fit_generator 可以继承keras.utils.Sequence来进行
### 中文keras网址 https://keras.io/zh/utils/#sequence
### -----> 好处：Sequence 是进行多进程处理的更安全的方法。这种结构保证网络在每个时期每个样本只训练一次，这与生成器不同。
### 指导网址：https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
### https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, data_X,data_Y, tok_list,max_lens,batch_size=1, shuffle=True):
        ### 这部分自己修改
        self.batch_size = batch_size
        self.data_X = data_X
        self.data_Y = data_Y
        self.indexes = np.arange(len(self.data_X))
        self.shuffle = shuffle
        self.tok_list = tok_list
        self.max_lens = max_lens
        self.on_epoch_end()

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.data_X) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取data_X集合中的数据
        batch_data_X = self.data_X.iloc[batch_indexs,:]
        batch_data_Y = self.data_Y[batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_data_X,batch_data_Y)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_data_X,batch_data_Y):
        ####这部分自己写
        X_train_list = []
        targets = ['creative_id','ad_id','advertiser_id','product_id','industry']
        for i,target in enumerate(targets):
            shuffled_target = batch_data_X['list_'+target].map(lambda x:list(np.random.permutation(x)))
            shuffled_target = self.tok_list[i].texts_to_sequences(shuffled_target)
            shuffled_target = pad_sequences(shuffled_target,maxlen = self.max_lens[i],value = 0)
            X_train_list.append(shuffled_target)
        #如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return X_train_list, batch_data_Y

model.fit_generator(train_generator,\
            validation_data  = ([X_test_cre2,X_test_ad2,X_test_adv2,X_test_prod2,X_test_industry2],y_test2), 
            epochs=100,verbose=1, callbacks=[es],workers = 4,use_multiprocessing=True)
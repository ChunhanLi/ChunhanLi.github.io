#### 文章 
#### https://mp.weixin.qq.com/s?__biz=MzIwNDA5NDYzNA==&mid=2247484742&idx=1&sn=a92b053058ba056c54c68b62234ad635&chksm=96c42883a1b3a195e7ea7ffcf53af950e221f5b24cac0cc66f9fd24705a35dfb6ed222d3fe21&scene=126&sessionid=1590396011&key=e4739a048b456af83dbf2361a30a36272155abc8921930b0457ed8394cd47ed1ba6f1b2407dca53c449842c9386dec942770c45fd8059a9e699f70ae9c008911d9a49980c7e4faa6b53bd15f4866c7f9&ascene=1&uin=MjE5NjM3MzgwMQ%3D%3D&devicetype=Windows+10+x64&version=62090070&lang=zh_CN&exportkey=AZr39lPSCGQAXow0v%2BFTipw%3D&pass_ticket=uyiWgrLxMlHuAjeYGFfVQ%2BN8MmnbKohhQnLSF63TYGg7L%2BFkDrzQu9wxYTEKi%2FQo

from gensim.models import Word2Vec
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
def word2vec_feature(prefix, df, groupby, target,size,window,min_count):
    df_bag = pd.DataFrame(df.sort_values('time')[[groupby, target]])####是否需要排序 不需要把.sort_values('time)去掉
    df_bag[target] = df_bag[target].astype(str)
    #df_bag[target].fillna('NAN', inplace=True)
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(set(x)))})####可以改list set
    doc_list = list(df_bag['list'].values)
    w2v = Word2Vec(doc_list, size=size, window=window, min_count=min_count, workers=32,sg=1,seed = 47,iter = 5)####iter默认是5
    vocab_keys = list(w2v.wv.vocab.keys())
    w2v_array = []
    for v in vocab_keys :
        w2v_array.append(list(w2v.wv[v]))
    df_w2v = pd.DataFrame()
    df_w2v['vocab_keys'] = vocab_keys    
    df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_array)], axis=1)
    df_w2v.columns = [target] + ['w2v_%s_%s_%d'%(prefix,target,x) for x in range(size)]
    print ('df_w2v:' + str(df_w2v.shape))
    return df_w2v


def w2v_model_2_df(w2v,prefix,target,size):
    """
    w2v:model
    predic/target:自己随便设定
    size: w2v的维度
    """
    vocab_keys = list(w2v.wv.vocab.keys())
    w2v_array = []
    for v in vocab_keys :
        w2v_array.append(list(w2v.wv[v]))
    df_w2v = pd.DataFrame()
    df_w2v['vocab_keys'] = vocab_keys    
    df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_array)], axis=1)
    df_w2v.columns = [target] + ['w2v_%s_%s_%d'%(prefix,target,x) for x in range(size)]
    print ('df_w2v:' + str(df_w2v.shape))
    return df_w2v


#####转化成mean/std...
def transform_to_feature_set(df,df_w2v,groupby,target):
    
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag[target] = df_bag[target].astype(str)
    #df_bag[target].fillna('NAN', inplace=True)
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(set(x)))})
    
    array_df_w2v = df_w2v.values
    index_mapper = df_w2v[[target]].to_dict()[target]
    index_mapper_inverse = {}
    for key,value in index_mapper.items():
        index_mapper_inverse[str(value)] = key
    list_mean = []
    list_std = []
    list_quantile25 = []
    list_quantile75 = []
    list_max = []
    list_min = []
    for index,val in tqdm(df_bag.iterrows(),total = df_bag.shape[0]):
        temp_list = val['list']
        temp_list_save = np.array(list(map(lambda x:index_mapper_inverse.get(str(x),-1),temp_list)))
        temp_list_save = list(temp_list_save[temp_list_save!=-1])
        val_all = array_df_w2v[temp_list_save,1:]
        val_all = np.array(val_all,dtype=np.float64)
        
        val_mean = val_all.mean(axis=0)
        list_mean.append(list(val_mean))
        
        val_std = val_all.std(axis=0)
        list_std.append(list(val_std))

        val_max = val_all.max(axis=0)
        list_max.append(list(val_max))

        val_min = val_all.min(axis=0)
        list_min.append(list(val_min))
        
        val_quantile25 = np.quantile(val_all,0.25,axis=0)
        list_quantile25.append(list(val_quantile25))
                    
        val_quantile75 = np.quantile(val_all,0.75,axis=0)
        list_quantile75.append(list(val_quantile75))
        
    w2v_mean = pd.DataFrame(list_mean,columns=[_+'_mean' for _ in df_w2v.columns[1:]])
    w2v_std = pd.DataFrame(list_std,columns=[_+'_std' for _ in df_w2v.columns[1:]])
    w2v_max = pd.DataFrame(list_max,columns=[_+'_max' for _ in df_w2v.columns[1:]])
    w2v_min = pd.DataFrame(list_min,columns=[_+'_min' for _ in df_w2v.columns[1:]])
    w2v_quantile25 = pd.DataFrame(list_quantile25,columns=[_+'_quantile25' for _ in df_w2v.columns[1:]])
    w2v_quantile75 = pd.DataFrame(list_quantile75,columns=[_+'_quantile75' for _ in df_w2v.columns[1:]])
                            
    return w2v_mean,w2v_std,w2v_max,w2v_min,w2v_quantile25,w2v_quantile75
#### example
df_w2v_creative_id = word2vec_feature('d64w5m5',all_merge,'user_id','creative_id',128,50,5)
w2v_creative_id_mean,w2v_creative_id_std,w2v_creative_id_max,w2v_creative_id_min,\
                        w2v_creative_id_quantile25,w2v_creative_id_quantile75 \
                    = transform_to_feature_set(all_merge,df_w2v_creative_id,'user_id','creative_id')
w2v_creative_id_mean.to_pickle('w2v_creative_id_mean.pkl')
w2v_creative_id_std.to_pickle('w2v_creative_id_std.pkl')
w2v_creative_id_max.to_pickle('w2v_creative_id_max.pkl')
w2v_creative_id_min.to_pickle('w2v_creative_id_min.pkl')
w2v_creative_id_quantile25.to_pickle('w2v_creative_id_quantile25.pkl')
w2v_creative_id_quantile75.to_pickle('w2v_creative_id_quantile75.pkl')


#### another version
#### model是为w2v model
emb_matrix = []
for col in tqdm(data['stars'].values):
    tmp = np.zeros(shape=(8))
    for seq in col:
        tmp = tmp + model.wv[str(seq)] / len(col)
    emb_matrix.append(tmp)
emb_matrix = np.array(emb_matrix)

for i in range(8):
    data['{}_{}_{}'.format('did','stars',i)] = emb_matrix[:,i]

### 两组embedding 对推荐物品本身/和用户
def two_w2v_features(df,groupby,target,size,window,min_count,feature):
    print('begin train word2vec')
    df_bag = data[[groupby,target]].copy()
    df_bag[target] = df_bag[target].astype(str)
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(set(x)))})
    doc_list = list(df_bag['list'].values)
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    if os.path.exists(f'./w2v/w2v_get2_{groupby}_{target}_d{size}_w{window}_m{min_count}.model'):
        model = Word2Vec.load(f'./w2v/w2v_get2_{groupby}_{target}_d{size}_w{window}_m{min_count}.model')
    else:
        model = Word2Vec(doc_list, size=size, window=window,\
                                       min_count=min_count, workers=32,sg=1,seed = 47,iter = 5)####iter默认是5
        model.save(f'./w2v/w2v_get2_{groupby}_{target}_d{size}_w{window}_m{min_count}.model')
    emb_matrix = []
    emb_dict = {}
    print('begin make feature')
    for seq in tqdm(doc_list):
        vec = []
        for w in seq:
            if w in model:
                vec.append(model.wv[w])
                emb_dict[w] = model.wv[w]
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * size)
    emb_matrix = np.array(emb_matrix)
    for i in range(size):
        df_bag[f'w2v_{groupby}_{target}_{i}'] = emb_matrix[:, i]
        feature.append(f'w2v_{groupby}_{target}_{i}')
    new_emb_martix = []
    data_index = []
    for v in tqdm(emb_dict):
        data_index.append(v)
        tmp_emb = np.array(emb_dict[v])
        new_emb_martix.append(tmp_emb)
    new_emb_martix = np.array(new_emb_martix)
    df_bag1 = pd.DataFrame()
    df_bag1[target] = data_index
    for i in range(size):
        df_bag1[f'w2v_single_{target}_{i}'] = new_emb_martix[:,i]
        feature.append(f'w2v_single_{target}_{i}')
    return df_bag,feature,df_bag1

###诈骗电话赛
### 做w2v模型
doc_list = train_voc_201908.groupby(['phone_no_m'])['opposite_no_m'].agg(lambda x:list(set(x)))
for df in [train_voc_201909,train_voc_201910,train_voc_201911,train_voc_201912,train_voc_202001,\
          train_voc_202002,train_voc_202003,test_voc_chusai,test_voc_fusai,test_voc]:
    tmp = df.groupby(['phone_no_m'])['opposite_no_m'].agg(lambda x:list(set(x)))
    doc_list = doc_list.append(tmp,ignore_index=True)
doc_list = list(doc_list.reset_index()['opposite_no_m'].values)
from gensim.models import Word2Vec
import logging
import logging
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
w2v_model = Word2Vec(doc_list, size=20, window=50, min_count=3, workers=12,sg=1,seed = 47,iter = 5)####iter默认是5
w2v_model.save('app_w2v_size32_minc1_window500_iter10.model')
w2v = Word2Vec.load('oppo_w2v.model')

### w2v特征合并
    #### w2v app
    tmp_app = train_voc.groupby(['phone_no_m'])['opposite_no_m'].agg(lambda x:list(set(x)))
    emb_matrix = []
    for seq in tqdm(tmp_app):
        vec = []
        for w in seq:
            if w in w2v:
                vec.append(w2v.wv[w])
        if len(vec) >0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * 20)
    emb_matrix = np.array(emb_matrix)
    tmp1 = pd.DataFrame()
    tmp1['phone_no_m'] = tmp_app.index
    for i in range(20):
        tmp1[f'w2v_app_{i}'] = emb_matrix[:, i]
    combine_list.append(tmp1)
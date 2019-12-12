from scipy import sparse
import pandas as pd
import numpy as np
import os

import re

import tensorflow as tf
'''
from node2vec import Node2Vec
import node2vec as n2v
'''
import gensim

def load_n2v(n2v_file):
    vectors = gensim.models.KeyedVectors.load(n2v_file)
    return vectors

def load_item_info(csv_file):
    tp = pd.read_csv(csv_file, delimiter=r'::', names=['movieId', 'title','genres'], engine='python')
    
    return tp
    
def load_unique_sid(txt_file):
    f = open(txt_file,'r')
    data = f.read()
    f.close()
    data = data.strip().split('\n')
    return data

def load_user_info(csv_file, raw_flag = False):
    tp = pd.read_csv(csv_file, engine='python')
    
    if raw_flag:
        return tp
    gender = tp['gender'].apply([lambda x: 0 if x=='F' else 1])
    gender_df = pd.DataFrame(gender)
    gender_df.columns = ['gender']
    tp = tp.drop(columns=['gender'])
    tp = tp.merge(gender_df, left_index=True, right_index=True)
    age_dummy = pd.get_dummies(tp['age'], prefix='age')
    tp = tp.merge(age_dummy, left_index=True, right_index=True)
    tp = tp.drop(columns=['age'], axis=1)
    occupation_dummy = pd.get_dummies(tp['occupation'], prefix='occupation')
    tp = tp.merge(occupation_dummy, left_index=True, right_index=True)
    tp = tp.drop(columns=['occupation'])
    
    zip_code_db = pd.read_csv(os.path.join(csv_file[0:-10],r'free-zipcode-database.csv'), usecols=['Zipcode', 'State']).drop_duplicates()
    def zip_code_process(zip_code):
        code = zip_code[0:5]
        if re.search('[a-zA-Z]',code) != None:
            return 'other'
        code = int(zip_code[0:5])
        try:
            ans = zip_code_db.at[code,'State']
            return ans
        except KeyError:
            return 'other'
    
    tp.loc[:,'zip_code'] = tp['zip_code'].apply(zip_code_process)
    zip_code_dummy = pd.get_dummies(tp['zip_code'], prefix='zip_code')
    tp = tp.merge(zip_code_dummy, left_index=True, right_index=True)
    tp = tp.drop(columns=['zip_code'], axis=1)
    
    return tp

    
    

def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)),
                             dtype='float64',
                             shape=(n_users, n_items))
    if 'userId' in tp.columns:
        userId_map = {k: v for k, v in zip(pd.unique(tp['uid']), pd.unique(tp['userId']))}
        userId_map = {k:v for k,v in sorted(userId_map.items())}
        return data, userId_map
    return data, None


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)),
                                dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)),
                                dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    if 'userId' in tp_tr.columns and 'userId' in tp_te:
        userId_map = {k: v for k, v in zip(pd.unique(tp_tr['uid'])-start_idx, pd.unique(tp_tr['userId']))}
        userId_map = {k:v for k,v in sorted(userId_map.items())}
        return data_tr, data_te, userId_map
    return data_tr, data_te, None

    
def get_weighted_sum(n2v_vectors,vad_data, train_data, userId_map):
    
    total_vectors = []
    for i in range(vad_data.shape[0]):
        total_weight = 0
        sum_vector = None
        for j in range(train_data.shape[0]):
            weight = np.inner(vad_data[i],train_data[j]).item()
            if weight >0:
                total_weight+=weight
                if type(sum_vector) == type(None):
                    sum_vector = weight*n2v_vectors[str(userId_map[j])]
                else:
                    sum_vector+= weight*n2v_vectors[str(userId_map[j])]
        if total_weight == 0 :
            sum_vector = np.zeros_like(n2v_vectors[str(userId_map[0])])
        else :
            sum_vector = sum_vector / total_weight
        total_vectors.append(sum_vector)
    total_vectors = np.stack(total_vectors)
    return total_vectors
    
    
def tr_te_dataset(data_tr, data_te, batch_size, mode = "basic", user_info=None, vad_userId_map=None, n2v_vectors = None, train_data = None, userId_map=None):
    # https://www.tensorflow.org/performance/performance_guide makes me think that I'm doing
    # something wrong, because my GPU usage hovers near 0 usually. That's v disappointing. I hope
    # I can speed it up hugely...
    # This is going to take in the output of data_tr and data_te, and turn them into
    # things we can sample from.

    # The only worry I have is, I don't know exactly how to do the whole "masking" part in here..

    # The way it works is, load_train_data just loads in training data, while load_tr_te_data
    # has goal-vectors as well. These are the ones that you drop-out. So, this really should be fine.
    
    assert type(data_tr) != type(None)
    assert type(data_te) != type(None)
    assert type(batch_size) != type(None)

    data_tr = data_tr.astype(np.float32)
    data_tr_coo = data_tr.tocoo()

    data_te = data_te.astype(np.float32)
    data_te_coo = data_te.tocoo()
    
    n_items = data_tr_coo.shape[1]
    if mode == "one_hot":
        assert type(user_info) != type(None)
        assert type(vad_userId_map) != type(None)
        vad_user_info_cut = user_info[user_info['userId'].isin(vad_userId_map.values())]
        vad_user_info_cut = vad_user_info_cut.set_index(keys='userId')
        vad_user_info_matrix = vad_user_info_cut.loc[vad_userId_map.values()].values
        vad_user_info_matrix = vad_user_info_matrix.astype(np.float32)
        
        vad_data_tr = np.concatenate((data_tr.todense(), vad_user_info_matrix), axis=1)
        vad_data_tr = tf.convert_to_tensor(vad_data_tr)
        vad_data_te = np.concatenate((data_te.todense(), vad_user_info_matrix), axis=1)
        vad_data_te = tf.convert_to_tensor(vad_data_te)
        
        samples_tr = tf.data.Dataset.from_tensor_slices(vad_data_tr)
        samples_te = tf.data.Dataset.from_tensor_slices(vad_data_te)
        dataset = tf.data.Dataset.zip((samples_tr,samples_te)).shuffle(10000).batch(batch_size, drop_remainder=True)
        
        expected_shape = tf.TensorShape([batch_size, n_items+vad_user_info_cut.shape[1]])
    elif mode == "node2vec":
        assert type(n2v_vectors) != type(None)
        assert type(train_data) != type(None)
        assert type(userId_map) != type(None)
        n2v_tr = get_weighted_sum(n2v_vectors,data_tr.todense(), train_data.todense(), userId_map)
        vad_data_tr = np.concatenate((data_tr.todense(), n2v_tr), axis=1)
        vad_data_tr = tf.convert_to_tensor(vad_data_tr)
        n2v_te =  get_weighted_sum(n2v_vectors,data_te.todense(), train_data.todense(), userId_map)
        vad_data_te = np.concatenate((data_te.todense(), n2v_te), axis=1)
        vad_data_te = tf.convert_to_tensor(vad_data_te)
        
        samples_tr = tf.data.Dataset.from_tensor_slices(vad_data_tr)
        samples_te = tf.data.Dataset.from_tensor_slices(vad_data_te)
        dataset = tf.data.Dataset.zip((samples_tr,samples_te)).shuffle(10000).batch(batch_size, drop_remainder=True)
        
        expected_shape = tf.TensorShape([batch_size, n_items+n2v_vectors.vectors.shape[1]])
    elif mode == "node2vec_user_info":
        assert type(n2v_vectors) != type(None)
        assert type(user_info) != type(None)
        assert type(userId_map) != type(None)
        assert type(vad_userId_map) != type(None)
        user_info_cut = user_info[user_info['userId'].isin(userId_map.values())]
        user_info_cut = user_info_cut.set_index(keys='userId')
        user_info_matrix = user_info_cut.loc[userId_map.values()].values
        user_info_matrix = user_info_matrix.astype(np.float32)
        
        vad_user_info_cut = user_info[user_info['userId'].isin(vad_userId_map.values())]
        vad_user_info_cut = vad_user_info_cut.set_index(keys='userId')
        vad_user_info_matrix = vad_user_info_cut.loc[vad_userId_map.values()].values
        vad_user_info_matrix = vad_user_info_matrix.astype(np.float32)
        
        n2v = get_weighted_sum(n2v_vectors,vad_user_info_matrix, user_info_matrix, userId_map)
        vad_data_tr = np.concatenate((data_tr.todense(), n2v), axis=1)
        vad_data_tr = tf.convert_to_tensor(vad_data_tr)
        vad_data_te = np.concatenate((data_te.todense(), n2v), axis=1)
        vad_data_te = tf.convert_to_tensor(vad_data_te)
        
        samples_tr = tf.data.Dataset.from_tensor_slices(vad_data_tr)
        samples_te = tf.data.Dataset.from_tensor_slices(vad_data_te)
        dataset = tf.data.Dataset.zip((samples_tr,samples_te)).shuffle(10000).batch(batch_size, drop_remainder=True)
        
        expected_shape = tf.TensorShape([batch_size, n_items+n2v_vectors.vectors.shape[1]])
    else :
        indices = np.mat([data_tr_coo.row, data_tr_coo.col]).transpose()
        sparse_data_tr = tf.SparseTensor(indices, data_tr_coo.data, data_tr_coo.shape)


        indices = np.mat([data_te_coo.row, data_te_coo.col]).transpose()
        sparse_data_te = tf.SparseTensor(indices, data_te_coo.data, data_te_coo.shape)

        samples_tr = tf.data.Dataset.from_tensor_slices(sparse_data_tr)
        samples_te = tf.data.Dataset.from_tensor_slices(sparse_data_te)

        # 10000 might be too big to sample from... Not sure how that's supposed to work with batch anyways.
        dataset = tf.data.Dataset.zip((samples_tr, samples_te)).shuffle(10000).batch(
            batch_size, drop_remainder=True)

        dataset = dataset.map(lambda x, y: (tf.sparse_tensor_todense(x), tf.sparse_tensor_todense(y)))

        expected_shape = tf.TensorShape([batch_size, n_items])

    dataset = dataset.apply(tf.contrib.data.assert_element_shape((expected_shape, expected_shape)))

    # dataset = dataset.skip(15)

    return dataset
    # dataset = dataset.map()


def train_dataset(data_tr, batch_size, mode = "basic", user_info = None, userId_map = None, unique_sid = None, n2v_vectors = None):

    # Note: I'm going to do the most heinous of things: I'm going to add in a fake operation here,
    # so that it has the same form as the other guy.
    # That will let us swap them out.
   

    data_tr = data_tr.astype(np.float32)

    data_tr_coo = data_tr.tocoo()
    
 

    n_items = data_tr_coo.shape[1]

    if mode == "one_hot":
        user_info_cut = user_info[user_info['userId'].isin(userId_map.values())]
        user_info_cut = user_info_cut.set_index(keys='userId')
        user_info_matrix = user_info_cut.loc[userId_map.values()].values
        user_info_matrix = user_info_matrix.astype(np.float32)
        
        data = np.concatenate((data_tr.todense(), user_info_matrix), axis=1)
        data = tf.convert_to_tensor(data)
        samples_tr = tf.data.Dataset.from_tensor_slices(data)
        dataset = samples_tr.shuffle(10000).batch(batch_size, drop_remainder=True)
        expected_shape = tf.TensorShape([batch_size, n_items+user_info_cut.shape[1]])
    elif mode == "node2vec" or mode == "node2vec_user_info":
        data = np.concatenate((data_tr.todense(), n2v_vectors.vectors), axis=1)
        data = tf.convert_to_tensor(data)
        samples_tr = tf.data.Dataset.from_tensor_slices(data)
        dataset = samples_tr.shuffle(10000).batch(batch_size, drop_remainder=True)
        expected_shape = tf.TensorShape([batch_size, n_items+n2v_vectors.vectors.shape[1]])
    else:
        indices = np.mat([data_tr_coo.row, data_tr_coo.col]).transpose()
        sparse_data = tf.SparseTensor(indices, data_tr_coo.data, data_tr_coo.shape)

        samples_tr = tf.data.Dataset.from_tensor_slices(sparse_data)
        dataset = samples_tr.shuffle(10000).batch(batch_size, drop_remainder=True)#.map(tf.sparse_todense)
        dataset = dataset.map(tf.sparse_tensor_todense)
        expected_shape = tf.TensorShape([batch_size, n_items])



    dataset = dataset.apply(tf.contrib.data.assert_element_shape(expected_shape))

    dataset = dataset.zip((dataset, dataset))
    # dataset.apply(tf.contrib.data.assert_element_shape([expected_shape, expected_shape]))

    # dataset = dataset.skip(200)

    return dataset


def get_batch_from_list(idxlist, batch_size, batch_num, data):
    disc_training_indices = idxlist[(batch_size * batch_num):(batch_size * (batch_num + 1))]
    X_train = data[disc_training_indices]
    if sparse.isspmatrix(X_train):
        X_train = X_train.toarray()
    X_train = X_train.astype('float32')
    return X_train


def get_num_items(pro_dir):
    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)
    print("n_items: {}".format(n_items))
    return n_items

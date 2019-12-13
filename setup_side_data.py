from training import get_data
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import os
data_dict = get_data('ml-100k')

n_items = data_dict['n_items']
train_data = data_dict['train_data']
vad_data_tr = data_dict['vad_data_tr']
vad_data_te = data_dict['vad_data_te']
user_info = data_dict['user_info']
userId_map = data_dict['userId_map']
vad_userId_map = data_dict['vad_userId_map']
n2v_vectors = data_dict['n2v_vectors']

data_tr = train_data.astype(np.float32)
data_tr_dense = data_tr.todense()

my_graph1 = nx.Graph()
my_graph1.add_nodes_from(userId_map.values())

for i in range(data_tr_dense.shape[0]):
    for j in range(i+1,data_tr_dense.shape[0]):
        weightCK = np.inner(data_tr_dense[i],data_tr_dense[j]).item()
        if weightCK >0:
            my_graph1.add_edge(userId_map[i],userId_map[j],weight = weightCK)
            
node2vec1 = Node2Vec(graph=my_graph1, # target graph
                    dimensions=50, # embedding dimension
                    walk_length=10, # number of nodes in each walks 
                    p = 1, # return hyper parameter
                    q = 0.0001, # inout parameter, q값을 작게 하면 structural equivalence를 강조하는 형태로 학습됩니다. 
                    weight_key='weight', # if weight_key in attrdict 
                    num_walks=10, 
                    workers=2,
                    temp_folder = '/content/gdrive/My Drive/n2vtmp'
                   )

model1 = node2vec1.fit(window=2)

model1.wv.save(os.path.join('.','data','ml-100k','node2vec_model_vectors'))

user_info_cut = user_info[user_info['userId'].isin(userId_map.values())]

my_graph2 = nx.Graph()
my_graph2.add_nodes_from(userId_map.values())

for i in range(user_info_cut.values.shape[0]):
    for j in range(i+1,user_info_cut.values.shape[0]):
        weightCK = np.inner(user_info_cut.values[i],user_info_cut.values[j]).item()
        if weightCK >0:
            my_graph2.add_edge(userId_map[i],userId_map[j],weight = weightCK)
            
node2vec2 = Node2Vec(graph=my_graph2, # target graph
                    dimensions=50, # embedding dimension
                    walk_length=10, # number of nodes in each walks 
                    p = 1, # return hyper parameter
                    q = 0.0001, # inout parameter, q값을 작게 하면 structural equivalence를 강조하는 형태로 학습됩니다. 
                    weight_key='weight', # if weight_key in attrdict 
                    num_walks=10, 
                    workers=2,
                    temp_folder = '/content/gdrive/My Drive/n2vtmp'
                   )

model2 = node2vec2.fit(window=2)

model2.wv.save(os.path.join('.','data','ml-100k','node2vec_userInfo_vectors'))

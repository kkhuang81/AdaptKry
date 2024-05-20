import torch
import math
import numpy as np
import random
from dataset import load_nc_dataset
from collections import OrderedDict
import csv
import os
import sys
import time
from time import perf_counter

import scipy.sparse as sp
import torch
import pickle as pkl
import struct
import gc
import scipy.special as ss
from scipy.sparse import csr_matrix, coo_matrix

from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, is_undirected,dropout_adj

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_splits(labels, num_classes, percls_trn=20, val_lb=500, seed=12591):   
    
    num_nodes=labels.shape[0]
    index=[i for i in range(0,num_nodes)]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    train_idx=np.array(train_idx)              
    rest_index = [i for i in index if i not in train_idx]
    val_idx=np.array(rnd_state.choice(rest_index,val_lb,replace=False))
    test_idx=np.array([i for i in rest_index if i not in val_idx])    

    return train_idx, val_idx, test_idx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def lazyP(adj, tau):
    adj=tau*adj+(1-tau)*sp.eye(adj.shape[0]) 
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    del row_sum
    gc.collect()    
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    t=time.time()
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)    
    adj=d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) 
    print('matrix multiplication time: ', time.time()-t)  
    return adj

def edgeindex_construct(edge_index, num_nodes, tau):     
    num_edges=edge_index[0].shape[0]
    data=np.array([1]*num_edges)
    adj=sp.coo_matrix((data, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)).tocsr()
    
    t=time.time()
    adj=lazyP(adj, tau)
    lazy_update_time = time.time()-t
    print('lazyupdate: ', lazy_update_time)

    t=time.time()
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    sparse_mx_time = time.time()-t
    print('sparse_mx: ', sparse_mx_time)

    return adj,lazy_update_time, sparse_mx_time


def load_dataset(dataset_name="genius", K=6, tau=0.5, self_loop=True):  
    features_time  = 0
    lazy_update_time = 0
    sparse_mx_time = 0

    dataset_str = 'data/' + dataset_name +'/'+dataset_name+'.npz'
    data = np.load(dataset_str)
    edge_index, feat, labels=data['edge_index'], data['feats'], data['labels'] 
    edge_index = torch.LongTensor(edge_index) 
    labels = torch.LongTensor(labels)

    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    edge_index = to_undirected(edge_index) 
    print('self_loop:', self_loop)
    if self_loop:
        print('add self loop: ', edge_index.shape)
        edge_index=remove_self_loops(edge_index)[0]
        edge_index=add_self_loops(edge_index)[0]
        print('after add self loop: ', edge_index.shape)
    
    num_nodes, dim=feat.shape
    num_classes=np.max(labels.numpy())+1
    
    feat=torch.FloatTensor(feat)
    LP, lazy_update_time, sparse_mx_time=edgeindex_construct(edge_index, num_nodes, tau)    
    t1 = time.time()            
    features=[feat]
    basis=feat    
    for i in range(1,K+1):
        basis=torch.spmm(LP, basis)         
        features.append(basis)
    features = torch.cat(features,1)
    features_time = time.time()-t1
    print('feat diffusion time: ', features_time)
    print('total time : ', features_time + lazy_update_time)
    print('total time with sparse: ', features_time + lazy_update_time + sparse_mx_time)
    print(features.shape)
    del basis, LP
    gc.collect()        
    return features, labels, dim, features_time + lazy_update_time+sparse_mx_time

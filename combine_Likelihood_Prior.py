#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:12:42 2020

@author: zijun.cui
"""

import numpy as np
import os
import pickle
from scipy.special import softmax
import matplotlib.pyplot as plt

'''
This is the file combing the likelihood score and the prior score
The following calculation is for RotatE + prior on FB15k-237
Others remain the same
'''
# prior model
seen_triples = pickle.load(open("./Prior-Model-with-Types/FB15k-237/processed_results/train_valid_triples.pkl","rb"))
[relation_head_type_set, relation_tail_type_set] = pickle.load(open('./Prior-Model-with-Types/FB15k-237/processed_results/train_relation_htset_weighted_removeCT_thres10.pkl','rb'))
relation_list_prior = list(relation_tail_type_set)
triple_score_prior,_ = pickle.load(open("./Prior-Model-with-Types/FB15k-237/test_relation_removeCT_thres10.pkl", 'rb'))

# likelihood model
triple_score_rotatE,_ = pickle.load(open("./Embedding-based-Models/RotatE/test_RotatE_FB15k-237.pkl", 'rb'))


with open(os.path.join('./Embedding-based-Models/RotatE/data/FB15k-237', 'relation2id.txt')) as fin:
    relation2id = dict()
    for line in fin:
        relation, rid = line.strip().split('\t')
        relation2id[relation] = int(rid)


def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
         

input_path = './Embedding-based-Models/RotatE/data/FB15k-237/test.txt'
triple_score = {}
triple_rank = []
triple_triples = []
hits = []
triple_rank_rotatE = []
hits_rotatE = []
triple_rank_prior = []

for i in range(10):
    hits.append([])
    
for i in range(10):
    hits_rotatE.append([])
    
relation_list = list(relation2id)
map_arr = np.zeros(237)
for j in range(len(relation_list_prior)):
    r = relation_list_prior[j]
    r_idx = relation2id.get(r)
    map_arr[j] = r_idx

map_arr = map_arr.astype(int)


with open(input_path) as f:
    for i, line in enumerate(f.readlines()):
        print('triple %d'%(i))
        e1, r_gt, e2 = line.strip().split()   
        score_rotatE = triple_score_rotatE.get(i)      
        score_rotatE_map = score_rotatE[map_arr]
        score_rotatE_map = softmax(score_rotatE_map)
        score_prior = np.array(triple_score_prior.get(i))
        score_prior = score_prior/sum(score_prior)
        
        score = score_prior*score_rotatE_map
        idx = relation_list_prior.index(r_gt)
        
        
        sort_score = np.argsort(np.asarray(score))[::-1]
        rank = np.where(sort_score==idx)[0]
        
        sort_score_rotatE = np.argsort(score_rotatE_map)[::-1]
        rank_rotatE = np.where(sort_score_rotatE==idx)[0]
        
        sort_score_prior = np.argsort(score_prior)[::-1]
        rank_prior = np.where(sort_score_prior==idx)[0]
        
        
        triple_rank.append(rank+1)
            
        triple_rank_prior.append(rank_prior+1)
        triple_rank_rotatE.append(rank_rotatE+1)
        
        for hits_level in range(10):
            if rank <= hits_level:
                hits[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
            if rank_rotatE <= hits_level:
                hits_rotatE[hits_level].append(1.0)
            else:
                hits_rotatE[hits_level].append(0.0)
            
                
        triple_triples.append([e1,e2,r_gt])


print('combined')       
for i in range(10):
    print('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
print('Mean rank: {0}', np.mean(triple_rank))
print('Mean reciprocal rank: {0}', np.mean(1./np.array(triple_rank)))

print('Embedding')       
for i in range(10):
    print('Hits @{0}: {1}'.format(i+1, np.mean(hits_rotatE[i])))
print('Mean rank: {0}', np.mean(triple_rank_rotatE))
print('Mean reciprocal rank: {0}', np.mean(1./np.array(triple_rank_rotatE)))


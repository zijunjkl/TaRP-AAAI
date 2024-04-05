#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:49:36 2020

@author: zijun.cui
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


'''
This is the file to compute prior scores
'''

[relation_head_type_set, relation_tail_type_set] = pickle.load(open('./processed_results/train_relation_htset_weighted_removeCT_thres20.pkl','rb'))
entity_type_set = pickle.load(open("./processed_results/entity_type_set_softmax_remove_commontopic.pkl","rb"))

seen_triples = pickle.load(open("./processed_results/train_valid_triples.pkl","rb"))
input_path = './data/test.txt'
triple_score = {}
triple_rank = []
triple_triples = []
hits = []
for i in range(10):
    hits.append([])
entity_list = list(entity_type_set)
relation_list = list(relation_tail_type_set)
rhead_type_set_size = []
rtail_type_set_size = []
import time


with open(input_path) as f:
    for i, line in enumerate(f.readlines()):
        e1, e2, r_gt = line.strip().split()   
        e1_type_set = [e[0] for e in entity_type_set.get(e1)] 
        e2_type_set = [e[0] for e in entity_type_set.get(e2)] 
        print('triple %d'%(i))
        score = []
        for j in range(len(relation_list)):
            r = relation_list[j]
            triples_list = seen_triples.get(r)
            rhead_type_set = relation_head_type_set.get(r)[0]
            rhead_type_set_weights = relation_head_type_set.get(r)[1]
            rhead_type_set_size.append(len(rhead_type_set))
            
            rtail_type_set = relation_tail_type_set.get(r)[0]
            rtail_type_set_weights = relation_tail_type_set.get(r)[1]
            rtail_type_set_size.append(len(rtail_type_set))
            if not [e1, e2] in triples_list and not r == r_gt: #rest of candidate triples
                similarity_rh = sum(rhead_type_set_weights[intersection(rhead_type_set, e1_type_set)])/sum(rhead_type_set_weights)
                similarity_rt = sum(rtail_type_set_weights[intersection(rtail_type_set, e2_type_set)])/sum(rtail_type_set_weights)
                score.append(similarity_rt*similarity_rt)
            elif r == r_gt:           
                similarity_rh = sum(rhead_type_set_weights[intersection(rhead_type_set, e1_type_set)])/sum(rhead_type_set_weights)
                similarity_rt = sum(rtail_type_set_weights[intersection(rtail_type_set, e2_type_set)])/sum(rtail_type_set_weights)
                idx_score = similarity_rt*similarity_rt
                score.append(idx_score)
                idx = j
            else:
                score.append(0)
                     
        sort_score = np.argsort(np.asarray(score))[::-1]
        rank = np.where(sort_score==idx)[0]
        triple_rank.append(rank+1)
        
        for hits_level in range(10):
            if rank <= hits_level:
                hits[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
    
            
        triple_score[i] = score
        
for i in range(10):
    print('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
print('Mean rank: {0}', np.mean(triple_rank))
print('Mean reciprocal rank: {0}', np.mean(1./np.array(triple_rank)))

f = open("test_relation_F+H_removeCT_invFreq_thres20.pkl", 'wb')
pickle.dump([triple_score, triple_rank], f)
f.close()

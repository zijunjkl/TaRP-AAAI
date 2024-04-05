#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:04:38 2020

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
This is the file to remove noisy types with specifies threshold
'''

relation_tail_type_set = pickle.load(open("./processed_results/train_relation_tail_type_set_weighted_remove_commontopic.pkl",'rb'))
relation_head_type_set = pickle.load(open("./processed_results/train_relation_head_type_set_weighted_remove_commontopic.pkl",'rb'))

seen_triples = pickle.load(open("./processed_results/train_valid_triples.pkl","rb"))

percen = 0.1 ## threshold for removing noisy types
relation_list = list(relation_tail_type_set)

tail_type_set = {}
head_type_set = {}


for j in range(len(relation_list)):
    r = relation_list[j]
    triples_list = seen_triples.get(r)
    if r in relation_head_type_set:
        temp_set = relation_head_type_set.get(r)
        ceiling = max(temp_set)
        floor = min(temp_set[np.nonzero(temp_set)])
        threshold = floor + percen*(ceiling-floor)
        rhead_type_set = [j for j,v in enumerate(temp_set) if v >= threshold]
        rhead_type_set_weights = temp_set.copy()
        rhead_type_set_weights[temp_set<threshold] = 0
        if len(rhead_type_set) == 0:
            rhead_type_set = [j for j,v in enumerate(temp_set) if v >= 0]
            rhead_type_set_weights = temp_set.copy()
        
    if r in relation_tail_type_set:
        temp_set = relation_tail_type_set.get(r)
        ceiling = max(temp_set)
        floor = min(temp_set[np.nonzero(temp_set)])
        threshold = floor + percen*(ceiling-floor)
        rtail_type_set = [j for j,v in enumerate(temp_set) if v >= threshold]
        rtail_type_set_weights = temp_set.copy()
        rtail_type_set_weights[temp_set<threshold] = 0
        if len(rtail_type_set) == 0:
            rtail_type_set = [j for j,v in enumerate(temp_set) if v >= 0]
            rtail_type_set_weights = temp_set.copy()
    
    tail_type_set[r] = [rtail_type_set, rtail_type_set_weights]
    head_type_set[r] = [rhead_type_set, rhead_type_set_weights]


f = open("./processed_results/train_relation_htset_weighted_removeCT_thres10.pkl", 'wb')
pickle.dump([head_type_set, tail_type_set], f)
f.close()            

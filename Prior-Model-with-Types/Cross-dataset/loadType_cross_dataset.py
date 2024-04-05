#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:46:17 2020

@author: zijun.cui
"""

import collections
import numpy as np
import os
import pickle
import scipy.io as io
from scipy.special import softmax



def compute_relation_type_set_weighted(relation_specific_entity_set, entity_type_set, type2id):
    relation_type_set = {}
    relation_list = list(relation_specific_entity_set)
    for i in range(len(relation_list)):
        r = relation_list[i]
        e_set = relation_specific_entity_set.get(r)
        if isinstance(e_set, str):
            e_set = [e_set]
        print('relation %d: %d entities'%(i, len(e_set)))
        type_set = np.zeros(len(type2id))
        for j in range(len(e_set)):
            e = e_set[j]
            if e in entity_type_set:
                temp_list = entity_type_set.get(e)
                for k in temp_list:
                    type_set[k[0]] = type_set[k[0]]+k[1]
        if not sum(type_set) == 0:
            relation_type_set[r] = type_set     
        else:
            print('empty set')
            relation_type_set[r] = type_set
                        
            
    return relation_type_set


entity_match, entity_type_set_orig = pickle.load(open("./precessed_results/entity_typeset_DBfromFB.pkl", 'rb'))

entity_list = list(entity_match)
entity_type_set = {}
for i in np.arange(len(entity_list)):
    e = entity_list[i]
    if entity_match.get(e)[1] == 1:
        entity_type_set[e] = entity_type_set_orig.get(e)
        
f = open("./precessed_results/entity_typeset_2.pkl", 'wb')
pickle.dump(entity_type_set, f)
f.close()
        
type2id = pickle.load(open("./precessed_results/type2id.pkl", 'rb'))
relation_head, relation_tail = pickle.load(open("../dbpedia/precessed_results/train_relation_head_tail_entities.pkl",'rb'))

relation_head_type_set = compute_relation_type_set_weighted(relation_head, entity_type_set, type2id)
f = open("./precessed_results/train_relation_head_type_set_v2.pkl", 'wb')
pickle.dump(relation_head_type_set, f)
f.close()
relation_tail_type_set = compute_relation_type_set_weighted(relation_tail, entity_type_set, type2id)
f = open("./precessed_results/train_relation_tail_type_set_v2.pkl", 'wb')
pickle.dump(relation_tail_type_set, f)
f.close()
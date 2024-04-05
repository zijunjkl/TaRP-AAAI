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

def entity_type_sets_softmax(entity_type):
    entity_list = list(entity_type)
    type_list = {}
    entity_type_idx = {}
    count = 0
    size_t = 0
    for i in range(len(entity_type)):
        print(i)
        e = entity_list[i] 
        e_type_list = entity_type.get(e)
        if len(e_type_list) == 0:
            print('entity %s has no type'%e)
        e_type_list_idx = [] #type list for entity e [typeID, typeWeight]
        for j in range(len(e_type_list)):
            ele_orig = e_type_list[j].split('/')
            ele = []
            for ee in ele_orig:
                if not ee in ele:
                    ele.append(ee)
                else:
                    break
            
            num_ele = len(ele)
            weights = softmax(np.arange(num_ele))
            weights = weights[::-1]
            for idx in range(num_ele):
                temp = ele[idx]
                if not temp in type_list:
                    type_list[temp] = count
                    count = count + 1
                val = type_list.get(temp) #id of the type
                if not any([extra[0]==val for extra in e_type_list_idx]):
                    e_type_list_idx.append([val, weights[idx]])
                else:
                    pos = [extra[0]==val for extra in e_type_list_idx].index(True)
                    e_type_list_idx[pos][1] = min(e_type_list_idx[pos][1], weights[idx]) #v2
        entity_type_idx[e] = e_type_list_idx
        size_t = size_t + len(e_type_list_idx)
    
    print('average size = %f'%(size_t/len(entity_list)))
    return entity_type_idx, type_list


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
            
    return relation_type_set


input_path = './data/db_onto_small_mini.txt'
index_onto = {}
with open(input_path) as f:
    for i, line in enumerate(f.readlines()):
        e1, rel, e2 = line.strip().split()
        if rel == 'isa' and not e1 == e2:
            if not e1 in index_onto:
                index_onto[e1] = [e2]
            else:
                if isinstance(index_onto[e1], str) and not e2 == index_onto[e1]:
                    index_onto[e1] = [index_onto[e1],e2]
                elif not e2 in index_onto[e1]:
                    index_onto[e1].append(e2)
                    

input_path = './data/db_InsType_mini.txt'
InsType_train = {}
with open(input_path) as f:
    for i, line in enumerate(f.readlines()):
        e1, rel, e2 = line.strip().split()
        if not e1 in InsType_train:
            InsType_train[e1] = [e2]
        else:
            if isinstance(InsType_train[e1], str) and not e2 == InsType_train[e1]:
                InsType_train[e1] = [InsType_train[e1],e2]
            elif not e2 in InsType_train[e1]:
                InsType_train[e1].append(e2)
                

entity_list = list(InsType_train)
type2id = {}
idx = 0
entity_type_list = {}
for i in range(len(InsType_train)):
    entity = entity_list[i]
    type_list = []
    type__ = InsType_train.get(entity)
    if isinstance(type__, str):
        type__ = [type__]
        
    for type_ in type__:  #T0
        if type_ in index_onto: #T1
            for type_l1 in index_onto[type_]:
                if type_l1 in index_onto: #T2
                    for type_l2 in index_onto[type_l1]:
                        if type_l2 in index_onto: #T3
                            for type_l3 in index_onto[type_l2]:
                                if type_l3 in index_onto:
                                    for type_l4 in index_onto[type_l3]:
                                        if type_l4 in index_onto:
                                            for type_l5 in index_onto[type_l4]:                                                 
                                                temp = str(type_)+'/'+str(type_l1)+'/'+str(type_l2)+'/'+str(type_l3)+'/'+str(type_l4)+'/'+str(type_l5)
                                                type_list.append(temp) 
                                        else:                                            
                                            temp = str(type_)+'/'+str(type_l1)+'/'+str(type_l2)+'/'+str(type_l3)+'/'+str(type_l4)
                                            type_list.append(temp)                                               
                                else:
                                    temp = str(type_)+'/'+str(type_l1)+'/'+str(type_l2)+'/'+str(type_l3)
                                    type_list.append(temp)                                    
                        else:
                            temp = str(type_)+'/'+str(type_l1)+'/'+str(type_l2)
                            type_list.append(temp)
                else:
                    temp = str(type_)+'/'+str(type_l1)
                    type_list.append(temp)
        else:
             type_list.append(type_)   
        
    entity_type_list[entity] = type_list


entity_type_set, type2id = entity_type_sets_softmax(entity_type_list)

f = open("./precessed_results/entity_type_set_v2.pkl", 'wb')
pickle.dump(entity_type_set, f)
f.close()

f = open("./precessed_results/type2id.pkl", 'wb')
pickle.dump(type2id, f)
f.close()

relation_head, relation_tail = pickle.load(open("./precessed_results/train_relation_head_tail_entities.pkl",'rb'))

relation_head_type_set = compute_relation_type_set_weighted(relation_head, entity_type_set, type2id)
f = open("./precessed_results/train_relation_head_type_set_v2.pkl", 'wb')
pickle.dump(relation_head_type_set, f)
f.close()
relation_tail_type_set = compute_relation_type_set_weighted(relation_tail, entity_type_set, type2id)
f = open("./precessed_results/train_relation_tail_type_set_v2.pkl", 'wb')
pickle.dump(relation_tail_type_set, f)
f.close()
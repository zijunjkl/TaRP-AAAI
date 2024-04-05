#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:29:46 2020

@author: zijun.cui
"""

import collections
import numpy as np
import os
import pickle
import scipy.io as io
from scipy.special import softmax

def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
         
def load_index(input_path):
    index = {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, idx = line.strip().split()
            index[v] = idx
    return index

def load_triple(input_path):
    '''
    categorize by relation
    '''
    relation_head, relation_tail = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            e1, e2, r = line.strip().split()
            if r in relation_head:
                if isinstance(relation_head[r], str):
                    if not e1 == relation_head[r]:
                        relation_head[r] = [relation_head[r],e1]
                else:
                    if not e1 in relation_head[r]:
                        relation_head[r].append(e1)
            else:
                relation_head[r] = e1
                
            if r in relation_tail:
                if isinstance(relation_tail[r], str):
                    if not e2 == relation_tail[r]:
                        relation_tail[r] = [relation_tail[r],e2]
                else:
                    if not e2 in relation_tail[r]:
                        relation_tail[r].append(e2)
            else:
                relation_tail[r] = e2 
    return relation_head, relation_tail

def process_seen_triples(input_path1, input_path2):
    '''
    categorize by relation
    '''
    relation_triples = {}
    with open(input_path1) as f:
        for i, line in enumerate(f.readlines()):
            e1, e2, r = line.strip().split()
            if r in relation_triples:
                relation_triples[r].append([e1,e2])
            else:
                relation_triples[r] = [[e1,e2]]
                
    with open(input_path2) as f:
        for i, line in enumerate(f.readlines()):
            e1, e2, r = line.strip().split()
            if r in relation_triples:
                relation_triples[r].append([e1,e2])
            else:
                relation_triples[r] = [[e1,e2]]
    return relation_triples

def load_triple_entity(input_path):
    '''
    categorize by head entity
    '''
    entity_tail = {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            e1, e2, r = line.strip().split()
                
            if e1 in entity_tail:
                if isinstance(entity_tail[e1], str):
                    if not e2 == entity_tail[e1]:
                        entity_tail[e1] = [entity_tail[e1],e2]
                else:
                    if not e2 in entity_tail[e1]:
                        entity_tail[e1].append(e2)
            else:
                entity_tail[e1] = e2 
                
    return entity_tail


def load_entity_type(input_path):
    index = {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            temp = line.strip().split()
            v = temp[0]
            rest = temp[1:]
            index[v] = rest
    return index



def entity_type_sets_softmax(entity_type):
    entity_list = list(entity_type)
    type_list = {}
    entity_type_idx = {}
    count = 0
    size_t = 0
    for i in range(len(entity_list)):
        print(i)
        e = entity_list[i] 
        e_type_list = entity_type.get(e)
        if len(e_type_list) == 0:
            print('entity %s has no type'%e)
        e_type_list_idx = [] #type list for entity e [typeID, typeWeight]
        for j in range(len(e_type_list)):
            if e_type_list[j][0] == '/':
                temp_type_list = e_type_list[j][1:]
            else:
                temp_type_list = e_type_list[j]
            if not temp_type_list == 'common/topic':
                ele = temp_type_list.split('/')      
                num_ele = len(ele)
                weights = softmax(np.arange(num_ele))
                # weights = np.ones(num_ele)
                for idx in range(num_ele):
                    temp = ele[idx]
                    if temp == 'common' or temp == 'topic' or temp == '':
                        print(temp_type_list)
                    if not temp in type_list:
                        type_list[temp] = count
                        count = count + 1
                    val = type_list.get(temp) #id of the type
                    if not any([extra[0]==val for extra in e_type_list_idx]):
                        e_type_list_idx.append([val, weights[idx]])
                    else:
                        pos = [extra[0]==val for extra in e_type_list_idx].index(True)
#                        e_type_list_idx[pos][1] = e_type_list_idx[pos][1] + weights[idx]
                        e_type_list_idx[pos][1] = min(e_type_list_idx[pos][1], weights[idx]) #v2
        entity_type_idx[e] = e_type_list_idx
        size_t = size_t + len(e_type_list_idx)
        
    print('average type size = %f'%(size_t/len(entity_list)))
    return entity_type_idx, type_list


def compute_relation_type_set_weighted(relation_specific_entity_set, entity_type_set):
    relation_type_set = {}
    relation_list = list(relation_specific_entity_set)
    for i in range(len(relation_list)):
        r = relation_list[i]
        e_set = relation_specific_entity_set.get(r)
        if isinstance(e_set, str):
            e_set = [e_set]
        print('relation %d: %d entities'%(i, len(e_set)))
        type_set = np.zeros(631)
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

def compute_entity_type_set_weighted(entity_tail_set, entity_type_set):
    entity_tail_type_set = {}
    entity_list = list(entity_tail_set)
    for i in range(len(entity_list)):
        e = entity_list[i]
        e_set = entity_tail_set.get(e)
        if isinstance(e_set, str):
            e_set = [e_set]
        print('head entity %d: %d entities'%(i, len(e_set)))
        type_set = np.zeros(631)
        for j in range(len(e_set)):
            ee = e_set[j]
            if ee in entity_type_set:
                temp_list = entity_type_set.get(ee)
                for k in temp_list:
                    type_set[k[0]] = type_set[k[0]]+k[1]
        if not sum(type_set) == 0:
            entity_tail_type_set[e] = type_set  
        else:
            print('empty set')
    return entity_tail_type_set

'''
This is the file to process all the required type sets for prior score computation
'''

entity2id = load_index('./data/entity2id.txt')
relation2id = load_index('./data/relation2id.txt')
entity_type_418 = load_entity_type('./data/entity_type_418.txt')
newType = load_entity_type('./data/newType.txt')
entity_type_fb15k = load_entity_type('./data/entity2type.txt')
entity_list = list(entity2id)
entity_type = {}
for i in np.arange(len(entity2id)):
    e = entity_list[i]
    if e in entity_type_fb15k:
        entity_type[e] = entity_type_fb15k.get(e)
    elif e in entity_type_418:
        entity_type[e] = entity_type_418.get(e)
    elif e in newType:
        entity_type[e] = newType.get(e)
    else:
        print('missing')


entity_type_set, type2id = entity_type_sets_softmax(entity_type)
f = open("./processed_results/entity_type_set_softmax_remove_commontopic.pkl", 'wb')
pickle.dump(entity_type_set, f)
f.close()

f = open("./processed_results/type2id.pkl", 'wb')
pickle.dump(type2id, f)
f.close()

#entity_tail = pickle.load(open("train_entity_tail.pkl",'rb'))
#missing = list(set(list(entity2id)) - set(entity_tail))
#for i in np.arange(len(missing)):
#    e = missing[i]
#    entity_tail[e] = e
#

relation_tail = pickle.load(open("./processed_results/train_relation_tail.pkl",'rb'))
relation_tail_type_set = compute_relation_type_set_weighted(relation_tail, entity_type_set)
f = open("./processed_results/train_relation_tail_type_set_weighted_remove_commontopic.pkl", 'wb')
pickle.dump(relation_tail_type_set, f)
f.close()

relation_head = pickle.load(open("./processed_results/train_relation_head.pkl",'rb'))
relation_head_type_set = compute_relation_type_set_weighted(relation_head, entity_type_set)
f = open("./processed_results/train_relation_head_type_set_weighted_remove_commontopic.pkl", 'wb')
pickle.dump(relation_head_type_set, f)
f.close()


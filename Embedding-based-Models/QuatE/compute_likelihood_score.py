#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 19:13:47 2020

@author: zijun.cui
"""

import json
import numpy as np
import pickle
import os

def process_seen_triples(input_path1):
    '''
    categorize by relation
    '''
    relation_triples = {}
    with open(input_path1) as f:
        for i, line in enumerate(f.readlines()):
            e1, r, e2 = line.strip().split()
            if r in relation_triples:
                relation_triples[r].append([e1,e2])
            else:
                relation_triples[r] = [[e1,e2]]
    
                
    return relation_triples

def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 


def QuatE(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b):

    denominator_b = np.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
    s_b = s_b / denominator_b
    x_b = x_b / denominator_b
    y_b = y_b / denominator_b
    z_b = z_b / denominator_b

    A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
    B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
    C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
    D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

    score_r = (A * s_c + B * x_c + C * y_c + D * z_c)
    # print(score_r.size())
    # score_i = A * x_c + B * s_c + C * z_c - D * y_c
    # score_j = A * y_c - B * z_c + C * s_c + D * x_c
    # score_k = A * z_c + B * y_c - C * x_c + D * s_c
    return np.sum(score_r, -1)
        
'''
This file is to compute the likelihood score given learned embeddings learning by QuatE
In the following, we show the calculation on DBpedia as the example
The computation on other datasets remains the same
'''

path = './Embeddings-dbpedia/QuatE-4999.json' 

if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)

        emb_s_a = data.get('emb_s_a.weight')
        emb_x_a = data.get('emb_x_a.weight')
        emb_y_a = data.get('emb_y_a.weight')
        emb_z_a = data.get('emb_z_a.weight')
        
        rel_s_b = data.get('rel_s_b.weight')
        rel_x_b = data.get('rel_x_b.weight')
        rel_y_b = data.get('rel_y_b.weight')
        rel_z_b = data.get('rel_z_b.weight')
        
        seen_triples = process_seen_triples('./benchmarks/dbpedia/db_insnet_train.txt')
        
        entity2id = {}
        input_path = './benchmarks/dbpedia/entity2id.txt'
        with open(input_path) as f:
            for i, line in enumerate(f.readlines()):
                if i > 0:
                    e, idx = line.strip().split()   
                    entity2id[e] = int(idx)
        
        relation2id = {}
        input_path = './benchmarks/dbpedia/relation2id.txt'
        with open(input_path) as f:
            for i, line in enumerate(f.readlines()):
                if i > 0:
                    r, idx = line.strip().split()   
                    relation2id[r] = int(idx)       
        
        input_path = './benchmarks/dbpedia/db_insnet_test.txt'
        triple_score = {}
        triple_rank = []
        triple_triples = []
        hits = []
        for i in range(10):
            hits.append([])
        
        relation_list = list(relation2id)
        
        with open(input_path) as f:
            for i, line in enumerate(f.readlines()):
                print('triple %d'%(i))
                e1, r_gt, e2 = line.strip().split()   
                e1_idx = entity2id.get(e1)
                e2_idx = entity2id.get(e2)
                r_gt_idx = relation2id.get(r_gt)
                s_a = np.array(emb_s_a[e1_idx])
                x_a = np.array(emb_x_a[e1_idx])
                y_a = np.array(emb_y_a[e1_idx])
                z_a = np.array(emb_z_a[e1_idx])
        
                s_c = np.array(emb_s_a[e2_idx])
                x_c = np.array(emb_x_a[e2_idx])
                y_c = np.array(emb_y_a[e2_idx])
                z_c = np.array(emb_z_a[e2_idx])
        
                s_b = np.array(rel_s_b[0:len(relation_list)])
                x_b = np.array(rel_x_b[0:len(relation_list)])
                y_b = np.array(rel_y_b[0:len(relation_list)])
                z_b = np.array(rel_z_b[0:len(relation_list)])
                score = QuatE(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
                
                for j in range(len(relation_list)):
                    r = relation_list[j]
                    if r in seen_triples:
                        triples_list = seen_triples.get(r)
                    else:
                        triples_list = []
                    if not [e1, e2] in triples_list and not r == r_gt: #rest of candidate triples
                        continue
                    elif r == r_gt:           
                        idx = j
                    else:
                        score[j] = 0 # filter out
                             
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

f = open("QuatE_DBpedia.pkl", 'wb')
pickle.dump([triple_score, triple_rank], f)
f.close()
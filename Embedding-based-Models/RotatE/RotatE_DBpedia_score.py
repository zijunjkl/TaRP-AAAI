#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:11:49 2020

@author: zijun.cui
"""
import numpy as np
import os
import pickle

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


entity_embedding = np.load('./RotatE_DBpedia/entity_embedding.npy')
relation_embedding = np.load('./RotatE_DBpedia/relation_embedding.npy')

seen_triples = process_seen_triples('./data/dbpedia/train.txt')

with open(os.path.join('./data/dbpedia/', 'entities.txt')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join('./data/dbpedia/', 'relations.txt')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)


def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
         
def RotatE(head, relation, tail):
    pi = 3.14159265358979323846
    
    re_head = head[0:500]
    im_head = head[500:1000]
    re_tail = tail[0:500]
    im_tail = tail[500:1000]
    
    gamma = 24
    epsilon = 2
    hidden_dim = 500
    embedding_range = (gamma + epsilon) / hidden_dim
    phase_relation = relation/(embedding_range/pi)

    re_relation = np.cos(phase_relation)
    im_relation = np.sin(phase_relation)


    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    re_score = re_score - re_tail
    im_score = im_score - im_tail
    
    score = np.zeros([2,298,500]) #298 is the total number of relations
    score[0,:,:] = re_score
    score[1,:,:] = im_score
    score = np.linalg.norm(score, axis = 0)

    score = gamma - np.sum(score, axis=1)
    return score

'''
This file is to compute the likelihood score given learned embeddings learning by RotatE
In the following, we show the calculation on DBpedia as the example
The computation on other datasets remains the same
'''

input_path = './data/dbpedia/db_insnet_test.txt'
triple_score = {}
triple_rank = []
triple_triples = []
hits = []
for i in range(10):
    hits.append([])

relation_list = list(relation2id)
with open(input_path) as f:
    for i, line in enumerate(f.readlines()):
        print(i)
        e1, r_gt, e2 = line.strip().split()
        e1_idx = entity2id.get(e1)
        e1_embedding = entity_embedding[e1_idx, :]
        e2_idx = entity2id.get(e2)
        e2_embedding = entity_embedding[e2_idx, :]
        r_gt_idx = relation2id.get(r_gt)
        score = RotatE(e1_embedding, relation_embedding, e2_embedding)               
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

f = open("RotatE_DB_score.pkl", 'wb')
pickle.dump([triple_score, triple_rank], f)
f.close() 
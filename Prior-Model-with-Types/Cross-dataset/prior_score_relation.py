#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:55:58 2020

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
This is the file computing prior score on DBpedia with type collected from FB
'''

[relation_head_type_set, relation_tail_type_set] = pickle.load(open('./precessed_results/train_relation_htset_v2_thres0.pkl','rb'))
entity_type_set = pickle.load(open("./precessed_results/entity_typeset_2.pkl","rb"))

relation_list = pickle.load(open("./precessed_results/common_relation_list.pkl", 'rb'))
        
seen_triples = pickle.load(open("..dbpedia/precessed_results/train_triples.pkl","rb"))
input_path = '../dbpedia/data/db_insnet_test.txt'
triple_score = {}
triple_rank = []
triple_triples = []
hits = []
for i in range(10):
    hits.append([])
entity_list = list(entity_type_set)

rhead_type_set_size = []
rtail_type_set_size = []
unseen_relation = []
prior_flag = []
with open(input_path) as f:
    for i, line in enumerate(f.readlines()):
        e1, r_gt, e2 = line.strip().split()   
        if r_gt in relation_list:
            if e1 in entity_type_set and e2 in entity_type_set:
                e1_type_set = [e[0] for e in entity_type_set.get(e1)] 
                e2_type_set = [e[0] for e in entity_type_set.get(e2)] 
                print('triple %d'%(i))
                score = []
                flag = 2
                for j in range(len(relation_list)):
                    r = relation_list[j]
                    if r in seen_triples:
                        triples_list = seen_triples.get(r)
                    else:
                        triples_list = []
                    rhead_type_set = relation_head_type_set.get(r)[0]
                    rhead_type_set_weights = relation_head_type_set.get(r)[1]
                    
                    rtail_type_set = relation_tail_type_set.get(r)[0]
                    rtail_type_set_weights = relation_tail_type_set.get(r)[1]
                    if not [e1, e2] in triples_list and not r == r_gt: #rest of candidate triples
                        similarity_rh = sum(rhead_type_set_weights[intersection(rhead_type_set, e1_type_set)])/sum(rhead_type_set_weights)
                        similarity_rt = sum(rtail_type_set_weights[intersection(rtail_type_set, e2_type_set)])/sum(rtail_type_set_weights)
                        score.append(similarity_rh*similarity_rt)
                            
                    elif r == r_gt:           
                        similarity_rh = sum(rhead_type_set_weights[intersection(rhead_type_set, e1_type_set)])/sum(rhead_type_set_weights)
                        similarity_rt = sum(rtail_type_set_weights[intersection(rtail_type_set, e2_type_set)])/sum(rtail_type_set_weights)
                        idx_score = similarity_rh*similarity_rt
                        score.append(idx_score)
                        idx = j
                    else:
                        score.append(0)
 
                        
            elif (e1 in entity_type_set) and (e2 not in entity_type_set):
                e1_type_set = [e[0] for e in entity_type_set.get(e1)] 
                print('triple %d'%(i))
                score = []
                flag = 1
                for j in range(len(relation_list)):
                    r = relation_list[j]
                    if r in seen_triples:
                        triples_list = seen_triples.get(r)
                    else:
                        triples_list = []
                    rhead_type_set = relation_head_type_set.get(r)[0]
                    rhead_type_set_weights = relation_head_type_set.get(r)[1]
                    rhead_type_set_size.append(len(rhead_type_set))
                    
                    if not [e1, e2] in triples_list and not r == r_gt: #rest of candidate triples
                        similarity_rh = sum(rhead_type_set_weights[intersection(rhead_type_set, e1_type_set)])/sum(rhead_type_set_weights)
                        score.append(similarity_rh)
                    elif r == r_gt:           
                        similarity_rh = sum(rhead_type_set_weights[intersection(rhead_type_set, e1_type_set)])/sum(rhead_type_set_weights)
                        idx_score = similarity_rh
                        score.append(idx_score)
                        idx = j
                    else:
                        score.append(0)

                        
            elif (e2 in entity_type_set) and (e1 not in entity_type_set):
                e2_type_set = [e[0] for e in entity_type_set.get(e2)] 
                print('triple %d'%(i))
                score = []
                flag = 1
                for j in range(len(relation_list)):
                    r = relation_list[j]
                    if r in seen_triples:
                        triples_list = seen_triples.get(r)
                    else:
                        triples_list = []
                    rtail_type_set = relation_tail_type_set.get(r)[0]
                    rtail_type_set_weights = relation_tail_type_set.get(r)[1]
                    rtail_type_set_size.append(len(rtail_type_set))
                    
                    if not [e1, e2] in triples_list and not r == r_gt: #rest of candidate triples
                        similarity_rt = sum(rtail_type_set_weights[intersection(rtail_type_set, e2_type_set)])/sum(rtail_type_set_weights)
                        score.append(similarity_rt)
                    elif r == r_gt:           
                        similarity_rt = sum(rtail_type_set_weights[intersection(rtail_type_set, e2_type_set)])/sum(rtail_type_set_weights)
                        idx_score = similarity_rt
                        score.append(idx_score)
                        idx = j
                    else:
                        score.append(0)

                        
            else:
                flag = 0
                score = np.ones(len(relation_list)) # uniform with no prior
            
                    
        else:
            flag = -1
#            flag = 1
#            print('unseen relation %s'%r_gt)
#            unseen_relation.append(r_gt)
            score = np.ones(len(relation_list))
        
        if not flag == -1:
            sort_score = np.argsort(np.asarray(score))[::-1]
            rank = np.where(sort_score==idx)[0]
            triple_rank.append(rank[0]+1)
            
            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
    
        triple_score[i] = score
        prior_flag.append(flag)
        
           
        
# print('average size for e1:%f, for relation:%f'%(np.mean(np.array(rhead_type_set_size)), np.mean(np.array(rtail_type_set_size))))
        
for i in range(10):
    print('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
print('Mean rank: {0}'.format(np.mean(triple_rank)))
print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(triple_rank))))

f = open("test_prior_v2_thres0_flag2and1.pkl", 'wb')
pickle.dump([triple_score, prior_flag], f)
f.close()

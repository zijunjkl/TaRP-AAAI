#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:11:51 2020

@author: zijun.cui
"""

import numpy as np
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

#seen_triples = process_seen_triples('./train.txt')
#f = open("train_triples.pkl", 'wb')
#pickle.dump(seen_triples, f)
#f.close()
seen_triples = pickle.load(open("train_triples.pkl","rb"))

outF = open("train-80per.txt", "w")
perc = 0.8
relation_list = list(seen_triples)
for i in np.arange(len(relation_list)):
    rel = relation_list[i]
    triple_list = seen_triples.get(rel)
    idx = np.random.permutation(len(triple_list))
    num = int(perc*len(triple_list))
#    if len(triple_list) == 1:
#        num = len(triple_list)
#    elif len(triple_list) == 2 and perc < 0.5:
#        num = 1
#    elif len(triple_list) == 3 and perc == 0.2:
#        num = 1
#    elif perc == 0.05 and num == 0:
#        num = 1
    print('total triple = %d, subset = %d'%(len(triple_list), num))
    for j in np.arange(num):
        e1 = triple_list[idx[j]][0]
        e2 = triple_list[idx[j]][1]
        outF.write(str(e1)+'\t'+str(rel)+'\t'+str(e2))
        outF.write("\n")
    
#outF = open("train-80per.txt", "w")
#idx = np.random.permutation(len(triples_record))
#num = int(0.8*np.shape(triples_record)[0])
#for i in np.arange(num):
#    triple = triples_record[idx[i]]
#    outF.write(str(triple[0])+'\t'+str(triple[1])+'\t'+str(triple[2]))
#    outF.write("\n")
#outF.close()
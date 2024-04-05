#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:11:49 2020

@author: zijun.cui
"""
import numpy as np
import os
import pickle

import numpy as np
import pickle
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

def load_triples(filename, splitter = '@@@', line_end = '\n'):
    '''Load the dataset.'''
    triples = []
    # entity vocab
    ents = {}
    ent_tokens = {}
    # relation vocab
    rels = {}
    index_ents = {}
    index_rels = {}
    n_ents = 0
    n_rels = 0
    triples_record = set([])
    last_c = -1
    last_r = -1
    for line in open(filename):
        line = line.rstrip(line_end).split(splitter)
        if index_ents.get(line[0]) == None:
            last_c += 1
            ents[last_c] = line[0]
            index_ents[line[0]] = last_c
            ent_tokens[last_c] = set(line[0].replace('(','').replace(')','').split(' '))
        if index_ents.get(line[2]) == None:
            last_c += 1
            ents[last_c] = line[2]
            index_ents[line[2]] = last_c
            ent_tokens[last_c] = set(line[2].replace('(','').replace(')','').split(' '))
        if index_rels.get(line[1]) == None:
            last_r += 1
            rels[last_r] = line[1]
            index_rels[line[1]] = last_r
        h = line[0]
        r = line[1]
        t = line[2]
        triples.append([h, r, t])
#        triples_record.add((h, r, t))
    triples = np.array(triples)
    n_ents = last_c + 1
    n_rels = last_r + 1

    print("Loaded triples from", filename, ". #triples, #ents, #rels:", len(triples), n_ents, n_rels)
          
    return triples



#outF = open("entity2id.txt",'w')
#with open(os.path.join('./entities.txt')) as fin:
#    for line in fin:
#        eid, entity = line.strip().split('\t')
#        outF.write(entity + '\t' + eid)
#        outF.write("\n")
#outF.close()


#outF = open("relation2id.txt", 'w')
#with open(os.path.join('./relations.txt')) as fin:
#    for line in fin:
#        rid, relation = line.strip().split('\t')
#        outF.write(relation + "\t" + rid)
#        outF.write("\n")

with open(os.path.join('./entity2id.txt')) as fin:
    entity2id = dict()
    next(fin)
    for line in fin:
        entity, eid = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join('./relation2id.txt')) as fin:
    relation2id = dict()
    next(fin)
    for line in fin:
        relation, rid = line.strip().split('\t')
        relation2id[relation] = int(rid)


input_path = './db_insnet_train.txt'
triples_record = load_triples(input_path, splitter = '\t', line_end = '\n')

outF = open("train2id.txt", "w")
num = int(np.shape(triples_record)[0])
for i in np.arange(num):
    triple = triples_record[i]
    e1_idx = entity2id.get(triple[0])
    rel_idx = relation2id.get(triple[1])
    e2_idx = entity2id.get(triple[2])
    outF.write(str(e1_idx)+'\t'+str(e2_idx)+'\t'+str(rel_idx))
    outF.write("\n")
outF.close()
print('train=%d'%i)

outF = open("valid2id.txt", "w")
idx = np.random.permutation(len(triples_record))
num = int(0.1*np.shape(triples_record)[0])
for i in np.arange(num):
    triple = triples_record[idx[i]]
    e1_idx = entity2id.get(triple[0])
    rel_idx = relation2id.get(triple[1])
    e2_idx = entity2id.get(triple[2])
    outF.write(str(e1_idx)+'\t'+str(e2_idx)+'\t'+str(rel_idx))
    outF.write("\n")
outF.close()
print('valid=%d'%i)

input_path = './db_insnet_test.txt'
triples_record = load_triples(input_path, splitter = '\t', line_end = '\n')

outF = open("test2id.txt", "w")
num = int(np.shape(triples_record)[0])
for i in np.arange(num):
    triple = triples_record[i]
    e1_idx = entity2id.get(triple[0])
    rel_idx = relation2id.get(triple[1])
    e2_idx = entity2id.get(triple[2])
    outF.write(str(e1_idx)+'\t'+str(e2_idx)+'\t'+str(rel_idx))
    outF.write("\n")
outF.close()
print('test=%d'%i)
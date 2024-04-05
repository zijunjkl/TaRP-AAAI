#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:13:30 2020

@author: zijun.cui
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from difflib import get_close_matches
import difflib
import os


#entity_type_set_db = pickle.load(open("./entity_type_set_v2.pkl","rb")) # DBpedia
#entity_type_set_fb = pickle.load(open("./entity_type_set_softmax_remove_commontopic.pkl", 'rb')) #FreeBase
#entity_type_set_yg = pickle.load(open('./entity_type_set_normalize.pkl', 'rb')) #YAGO

source_data = 'FB'
entity_type_set_source = pickle.load(open("../FB15k/processed_results/entity_type_set_softmax_remove_commontopic.pkl", 'rb')) 
entity_list_source = list(entity_type_set_source)

entity_list_source_2 = pickle.load(open("./FB_entity_list.pkl", 'rb'))         

target_data = 'DB'
with open(os.path.join('../dbpedia/data/entities.txt')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

entity_list_target = list(entity2id)



entity_match_TfromS = {}
transfer_count = 0
entity_typeset_TfromS = {}
for i in np.arange(len(entity_list_target)):
    e = entity_list_target[i]
    closest = get_close_matches(e, entity_list_source_2, n=1)
    if not len(closest) == 0:
        transfer_count = transfer_count + 1
        idx = entity_list_source_2.index(closest[0])
        mid = entity_list_source[idx]
        entity_typeset_TfromS[e] = entity_type_set_source.get(mid)
        score = difflib.SequenceMatcher(None, e, closest[0]).ratio()
        print('%d entity, score=%f'%(i, score))
        entity_match_TfromS[e] = [closest, score]
    else:
        entity_typeset_TfromS[e] = []
        print('%d entity, score=%f'%(i, -999))
        entity_match_TfromS[e] = [[], -999]

#entity_typeset_yagofromDB = {}
#for e in entity_list_yg:
#    match = entity_match_yg.get(e)
#    if not len(match) == 0:
#        entity_typeset_yagofromDB[e] = entity_type_set_db.get(match[0])
#    else:
#        entity_typeset_yagofromDB[e] = []

file_string = 'entity_typeset_'+target_data+'from'+source_data+'.pkl'
f = open(file_string, 'wb')
pickle.dump([entity_match_TfromS, entity_typeset_TfromS], f)
f.close()    
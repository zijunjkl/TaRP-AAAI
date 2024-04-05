Folder <data> stores all the required data and types

Folder <processed_results> stores all the pre-computed type sets

File test_relation_removeCT_thres10.pkl is the pre-computed prior score which can be downloaded from the [link](https://www.dropbox.com/scl/fi/rxad06f2yfrrewffui28x/test_relation_removeCT_thres10.pkl?rlkey=w9dy8x03gqx7rr58ps29w39jt&dl=0)
To process all the required type sets for  prior score computation
Run `loadType_remove_common_topic.py`


To remove noisy types by specifying a threshold
Run `threshold_relation_type_set.py`

To compute the prior scores given obtained type sets
Run `prior_score_triple_relations.py`

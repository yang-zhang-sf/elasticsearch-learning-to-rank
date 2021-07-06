import xgboost as xgb

# Read the LibSVM labels/features
dtrain = xgb.DMatrix('xgboost.txt')
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear'}
# num_round = 2

# bst = xgb.train(param, dtrain, num_round)

# https://github.com/dmlc/xgboost/blob/b56d3d5d5c4d9fe5fe889e41fef1a7fe6e57bf0f/demo/rank/rank.py#L37-L40
params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
          'min_child_weight': 0.1, 'max_depth': 6}
bst = xgb.train(params, dtrain, num_boost_round=4)

model = bst.get_dump(fmap='featmap.txt', dump_format='json')

with open('xgb-model-ndcg.json', 'w') as output:
    output.write('[' + ','.join(list(model)) + ']')
    output.close()


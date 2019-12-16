# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import xgboost as xgb
import random
import time
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score, f1_score
import datetime as dt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# print(check_output(["ls", "../input"]).decode("utf8"))


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def get_features(train, test):
    trainval = list(train.columns.values)
    output = trainval
    return sorted(output)


def run_single(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 6
    subsample = 1
    colsample_bytree = 1
    min_chil_weight = 1
    start_time = time.time()

    print(
        'XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample,
                                                                                              colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "min_chil_weight": min_chil_weight,
        "seed": random_state,
        # "num_class" : 22,
    }
    num_boost_round = 500
    early_stopping_rounds = 20
    test_size = 0.1

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    gbm.save_model('0001.model')

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration + 1)
    print('check: %s' % check)

    # area under the precision-recall curve
    score = average_precision_score(X_valid[target].values, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

    check2 = check.round()
    print('check2: %s' % check2)

    score = precision_score(X_valid[target].values, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(X_valid[target].values, check2)
    print('recall score: {:.6f}'.format(score))

    score = accuracy_score(X_valid[target].values, check2)
    print('accurary score: {:.6f}'.format(score))

    score = f1_score(X_valid[target].values, check2)
    print('f1 score: {:.6f}'.format(score))

    RMSE = np.sqrt(mean_squared_error(X_valid[target].values, check2))
    print('RMSE:  {:.6f}'.format(RMSE.round(4)))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set... ")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration + 1)

    score = average_precision_score(test[target].values, test_prediction)
    print('area under the precision-recall curve test set: {:.6f}'.format(score))

    ############################################ ROC Curve

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(X_valid[target].values, check)
    roc_auc = auc(fpr, tpr)
    # xgb.plot_importance(gbm)
    # plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    ##################################################

    test['prob'] = test_prediction

    print('Training time: {} minutes'.format(round((time.time() - start_time) / 60, 2)))
    return test_prediction, imp, gbm.best_iteration + 1, test


# Any results you write to the current directory are saved as output.
start_time = dt.datetime.now()
print("Start time: ", start_time)

# data = pd.read_csv('../data/creditcard.csv')
data = pd.read_csv('data/Seasons_Stats.csv')
# data.loc[data['Player'] == 'LeBron James'].T
data = data.replace('?', np.nan)

zero_data = np.zeros(shape=(len(data), 1))
data = data.assign(all_nba=zero_data)

allnba = pd.read_csv('data/all_nba_teams_clean_binary.csv')
res = allnba.set_index(['Year', 'Player']) \
    .combine_first(data.set_index(['Year', 'Player'])) \
    .reset_index()
df_1955 = res[res.Year > 1975]
print(df_1955.Player == 'Michael Jordan')
# data = df_1955[['2P', '3P', '3P%', 'AST',
#                 'AST%', 'BLK', 'BLK%', 'DRB',
#                 'eFG%', 'FT', 'FT%', 'FTA',
#                 'G', 'MP', 'ORB', 'STL',
#                 'TOV', 'TOV%', 'TRB', 'TRB%', 'TS%', 'PTS', 'PER',
#                 'all_nba']].copy()

data = df_1955[['2P', '3P', 'eFG%', 'AST', 'BLK', 'FT', 'MP', 'ORB', 'DRB', 'TRB', 'STL', 'PTS', 'G', 'all_nba']].copy()

# data = df_1955[['all_nba']].copy()

train, test = train_test_split(data, test_size=.1, random_state=random.seed(2016))

test = df_1955[(df_1955.Player == 'Michael Jordan') |
               (df_1955.Player == 'Tim Duncan ') |
               (df_1955.Player == 'LeBron James') |
               (df_1955.Player == 'Dwyane Wade') |
               (df_1955.Player == 'Kevin Durant') |
               (df_1955.Player == 'Giannis Antetokounmpo') |
               (df_1955.Player == 'Russell Westbrook') |
               (df_1955.Player == 'Zach Randolph') |
               (df_1955.Player == 'Deron Williams') |
               (df_1955.Player == 'Vince Carter') |
               (df_1955.Player == 'Mike Muscala') |
               (df_1955.Player == 'Shane Battier') |
               (df_1955.Player == 'Mike Miller')
               ]
test = test[['Player', 'Year', '2P', '3P', 'eFG%', 'AST', 'BLK', 'FT', 'MP', 'ORB', 'DRB', 'TRB', 'STL', 'PTS', 'G',
             'all_nba']].copy()

features = list(train.columns.values)
features.remove('all_nba')
print(features)

print("Building model.. ", dt.datetime.now() - start_time)
preds, imp, num_boost_rounds, test_prob = run_single(train, test, features, 'all_nba', 42)

print(dt.datetime.now() - start_time)

print(test_prob.to_string())

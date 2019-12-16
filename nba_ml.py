import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/Seasons_Stats.csv')
# data.loc[data['Player'] == 'LeBron James'].T
data = data.replace('?', np.nan)

zero_data = np.zeros(shape=(len(data), 1))
data = data.assign(all_nba=zero_data)

allnba = pd.read_csv('data/all_nba_teams_clean.csv')
res = allnba.set_index(['Year', 'Player']) \
    .combine_first(data.set_index(['Year', 'Player'])) \
    .reset_index()
df_1955 = res[res.Year > 1955]
df_ml = df_1955[['2P', '2P%', '2PA', '3P', '3P%', '3PA', 'AST',
                 'AST%', 'BLK', 'BLK%', 'BPM', 'DRB', 'DRB%',
                 'eFG%', 'FG', 'FG%', 'FGA', 'FT', 'FT%', 'FTA',
                 'G', 'GS', 'MP', 'ORB', 'ORB%', 'PER', 'PF', 'STL',
                 'STL%', 'TOV', 'TOV%', 'TRB', 'TRB%', 'TS%',
                 'WS', 'WS/48']].copy()
df_anba = df_1955[['all_nba']].copy()

X_train, X_test, y_train, y_test = train_test_split(df_ml, df_anba, test_size=0.2, random_state=123)


params = {'learning_rate': 0.1, 'max_depth': 5, "objective": "reg:squarederror",
          'tree_method': 'gpu_hist'}
params['tree_method']='hist'
data_dmatrix = xgb.DMatrix(data=df_ml, label=df_anba)
gpu_res={}
# xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10, evals_result=gpu_res)

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10, predictor='cpu_predictor')

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)
print(preds)


rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

print(cv_results.head())

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]

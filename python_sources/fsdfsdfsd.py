import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import Lasso, BayesianRidge, ElasticNet, RANSACRegressor, HuberRegressor, Ridge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, IsolationForest
from sklearn.svm import OneClassSVM
from scipy.stats import skew

if __name__ == '__main__':
    def missing_values(df):
        df['SqrtLotArea'] = np.sqrt(df['LotArea'])
        cond = df['LotFrontage'].isnull()
        df.LotFrontage[cond] = df.SqrtLotArea[cond]
        df.loc[df['Alley'].isnull(), 'Alley'] = 'None'
        df.loc[df['MasVnrType'].isnull(), 'MasVnrType'] = 'None'
        df.loc[df['MasVnrArea'].isnull(), 'MasVnrArea'] = 0
        df.loc[df['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
        df.loc[df['FireplaceQu'].isnull(), 'FireplaceQu'] = 'None'
        to_delete = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'SqrtLotArea']
        df.loc[df['Fence'].isnull(), 'Fence'] = 'None'
        df.loc[df['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
        basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        for cols in basement_cols:
            df.loc[df[cols].isnull(), cols] = 'None'
        garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
        for cols in garage_cols:
            if df[cols].dtype == np.object:
                df.loc[df[cols].isnull(), cols] = 'None'
            else:
                df.loc[df[cols].isnull(), cols] = 0
        df.drop(to_delete, axis=1, inplace=True)
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index
        df[skewed_feats] = np.log1p(df[skewed_feats])


    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    outlier_idx =     [1349, 968, 462, 1324, 812, 218, 1328, 666, 688, 431, 335, 523, 137, 3, 970, 632, 495, 30, 1432, 1453, 628,
         313, 669, 1181, 365, 1119, 142, 607, 197, 271, 66, 613, 185, 17, 1170, 1211, 238, 1292, 874, 691, 457, 1182,
         574, 1298, 1383, 1186, 150, 190, 463, 112]
    train.drop(train.index[outlier_idx], inplace=True)
    missing_values(train)
    missing_values(test)
    train_id = train['Id']
    test_id = test['Id']
    all = pd.concat([train, test])
    # total = train.isnull().sum().sort_values(ascending=False)
    # percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
    # missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    # all = all.drop((missing_data[missing_data['Total'] > 1]).index, 1)
    # all = all.drop(all.loc[all['Electrical'].isnull()].index)
    result = pd.get_dummies(all)
    train = result.loc[result['Id'].isin(train_id)]
    test = result.loc[result['Id'].isin(test_id)]

    # X_train, X_test, y_train, y_test = train_test_split(train.drop(['SalePrice', 'Id'], axis=1), train['SalePrice'], test_size=0.2,random_state=0)
    # features = quantitative + boolean + qdr
    # regressor = GradientBoostingRegressor(n_estimators=1100, criterion='mse')
    #
    # regressor = RANSACRegressor(loss='squared_loss',max_trials=500)
    # re_list = [BayesianRidge(), RandomForestRegressor(random_state=0), ExtraTreesRegressor(random_state=0), HuberRegressor(),
    #            KNeighborsRegressor()]
    re_list = [RandomForestRegressor(n_jobs=4, random_state=0),
               ExtraTreesRegressor(n_jobs=4, random_state=0, ),
               GradientBoostingRegressor(random_state=0, ),
               ]
    # param_list = [
    #     {'n_iter': [100, 200, 300], 'tol': [1e-2, 1e-1], 'alpha_1': [1e-5, 1e-4], 'alpha_2': [1e-7, 1e-8],
    #      'lambda_1': [1e-7, 1e-8], 'lambda_2': [1e-5, 1e-4]},
    #     {'n_estimators': [700, 800, 900, 1000, 1100, 1200]},
    #     {'alpha': [0.5, 1.0, 1.5], 'tol': [1e-5, 1e-3, 1e-4]},
    #     {'n_estimators': [5, 10, 200, 500, 800, 900, 1000, 1100, 1200]},
    #     {'epsilon': [1.25, 1.35, 1.5], 'max_iter': [50, 100, 200, 500], 'alpha': [1e-5, 1e-3, 1e-4], 'tol': [1e-5, 1e-6, 1e-4]},
    #     {'n_neighbors': [2, 5, 10, 50, 100]},
    #     {'C': [0.5, 0.1, 1, 2, 5, 10], 'epsilon': [0.05, 0.1, 0.2], 'kernel': ['linear', 'rbf', 'sigmoid'],
    #      'tol': [1e-2, 1e-3, 1e-4]}]
    param_list = [
        {'n_estimators': [300, 400, 500, 600, 700, 800, 1000], 'max_features': [16, 17, 18, 19, 20, 'auto'],
         'max_depth': [9, 10, 11, 12, 13, None]},
        {'n_estimators': [300, 400, 500, 600, 700, 800, 1000], 'max_features': [18, 19, 20, 21, 22, 'auto'],
         'max_depth': [9, 10, 11, 12, 13, None]},
        {'n_estimators': [300, 400, 500, 600, 700, 800, 1000], 'max_features': [8, 9, 10, 11, 12, 'auto'],
         'max_depth': [4, 5, 6, 7, 8, None],
         'learning_rate': [0.2, 0.1, 0.05, 0.01], 'subsample': [0.8, 0.9, 1.0]},
    ]
    tr = train.drop(['SalePrice', 'Id'], axis=1)
    X = tr.fillna(0.).values
    Y = train['SalePrice'].values
    regressor = XGBRegressor(
    colsample_bytree=0.2,
    gamma=0.0,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1.5,
    n_estimators=7300,
    reg_alpha=0.9,
    reg_lambda=0.5,
    subsample=0.2,
    seed=42,
    silent=1)
    regressor.fit(X,Y)
    ids = test['Id']
    test = test.drop(['Id', 'SalePrice'], axis=1)
    Ypred = np.exp(regressor.predict(test.fillna(0.).values))
    sub = pd.DataFrame({
        "Id": ids,
        "SalePrice": Ypred
    })
    sub.to_csv("prices_submission.csv", index=False)
    # for i, regressor in enumerate(re_list):
    # gsCV = GridSearchCV(XGBRegressor(seed=0), param_grid={'n_estimators': [300, 400, 500, 600, 700, 800, 1000], 'max_depth': [3, 5, 6, 7, 8, 9],
    #  'learning_rate': [0.2, 0.1, 0.05, 0.01], 'subsample': [0.7,0.8, 0.9, 1.0], 'colsample_bytree': [0.5, 0.6, 0.75, 0.8, 1]}, scoring='neg_mean_squared_error', cv=4, n_jobs=-1)
    # gsCV.fit(X, Y)
    # print gsCV.best_params_
    # print gsCV.best_score_
    # best_res = [  # BayesianRidge(n_iter=100, lambda_1=1e-08, lambda_2=0.0001, alpha_2=1e-08, tol=0.1, alpha_1=0.0001),
    #     # RandomForestRegressor(n_estimators=1200, n_jobs=-1), ExtraTreesRegressor(n_estimators=1100, n_jobs=-1),
    #     # RANSACRegressor(loss='squared_loss', max_trials=500),
    #     # GradientBoostingRegressor(n_estimators=1100, criterion='mse'),
    #     RandomForestRegressor(
    #         n_jobs=-1, random_state=0,
    #         n_estimators=500, max_features=18, max_depth=11
    #     ),
    #     ExtraTreesRegressor(
    #         n_jobs=-1, random_state=0,
    #         n_estimators=500, max_features=20
    #     ),
    #     GradientBoostingRegressor(
    #         random_state=0,
    #         n_estimators=500, max_features=10, max_depth=6,
    #         learning_rate=0.05, subsample=0.8
    #     ),
    #     XGBRegressor(
    #         seed=0,
    #         n_estimators=500, max_depth=7,
    #         learning_rate=0.05, subsample=0.8, colsample_bytree=0.75
    #     ),
    #     RANSACRegressor(loss='squared_loss', max_trials=500)
    # ]
    #
    #
    # class Ensemble(object):
    #     def __init__(self, n_folds, stacker, base_models):
    #         self.n_folds = n_folds
    #         self.stacker = stacker
    #         self.base_models = base_models
    #
    #     def fit_predict(self, X, y, T):
    #         X = np.array(X)
    #         y = np.array(y)
    #         T = np.array(T)
    #         kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
    #         S_train = np.zeros((X.shape[0], len(self.base_models)))
    #         S_test = np.zeros((T.shape[0], len(self.base_models)))
    #         for i, clf in enumerate(self.base_models):
    #             S_test_i = np.zeros((T.shape[0], self.n_folds))
    #             for j, (train_idx, test_idx) in enumerate(kf.split(X)):
    #                 X_train = X[train_idx]
    #                 y_train = y[train_idx]
    #                 X_holdout = X[test_idx]
    #                 clf.fit(X_train, y_train)
    #                 y_pred = clf.predict(X_holdout)[:]
    #                 S_train[test_idx, i] = y_pred
    #                 S_test_i[:, j] = clf.predict(T)[:]
    #             S_test[:, i] = S_test_i.mean(1)
    #         param_grid = {
    #             'alpha_1': [1e-5, 1e-4], 'alpha_2': [1e-7, 1e-8],
    #             'lambda_1': [1e-7, 1e-8], 'lambda_2': [1e-5, 1e-4],
    #         }
    #         grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=-1, cv=4, scoring='neg_mean_squared_error')
    #         grid.fit(S_train, y)
    #         print('Best Params:')
    #         print(grid.best_params_)
    #         print('Best CV Score:')
    #         print(-grid.best_score_)
    #         y_pred = grid.predict(S_test)[:]
    #         return y_pred
    #
    #
    # ids = test['Id']
    # test = test.drop(['Id', 'SalePrice'], axis=1)
    # reg = Ensemble(5, BayesianRidge(), best_res)
    # Ypred = np.exp(reg.fit_predict(X, Y, test.fillna(0.).values))
    # sub = pd.DataFrame({
    #     "Id": ids,
    #     "SalePrice": Ypred
    # })
    # sub.to_csv("prices_submission.csv", index=False)

    # param_grid = {'loss': ['ls'],
    #               'n_estimators': [900,1000,1100],
    #               'criterion':['mse'],
    #               'max_depth':[2,3,4,5]}
    #
    # regressor=GradientBoostingRegressor(n_estimators=1100, criterion='mse')
    # regressor.fit(X,Y)
    # # CV_scores = np.sqrt(-cross_val_score(regressor, X, Y, n_jobs=-1, scoring='neg_mean_squared_error', cv=4))
    # # print CV_scores, np.mean(CV_scores)
    # ids = test['Id']
    # test = test.drop(['Id', 'SalePrice'], axis=1)
    # Ypred = np.exp(regressor.predict(test.fillna(0.).values))
    # sub = pd.DataFrame({
    #     "Id": ids,
    #     "SalePrice": Ypred
    # })
    # sub.to_csv("prices_submission.csv", index=False)

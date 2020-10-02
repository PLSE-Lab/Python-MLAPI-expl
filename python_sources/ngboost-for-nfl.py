#!/usr/bin/env python
# coding: utf-8

# Include a standalone version of NGBoost derived from git://github.com/stanfordmlgroup/ngboost.git

# In[ ]:


import numpy as np
import numpy.random as np_rnd
import scipy as sp
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import norm as dist

def default_tree_learner(depth=3):
    return DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=depth,
        splitter='best')

class MLE:
    def __init__(self, seed=123):
        pass

    def loss(self, forecast, Y):
        return forecast.nll(Y.squeeze()).mean()

    def grad(self, forecast, Y, natural=True):
        fisher = forecast.fisher_info()
        grad = forecast.D_nll(Y)
        if natural:
            grad = np.linalg.solve(fisher, grad)
        return grad


class CRPS:
    def __init__(self, K=32):
        self.K = K

    def loss(self, forecast, Y):
        return forecast.crps(Y.squeeze()).mean()

    def grad(self, forecast, Y, natural=True):
        metric = forecast.crps_metric()
        grad = forecast.D_crps(Y)
        if natural:
            grad = np.linalg.solve(metric, grad)
        return grad

EPS = 1e-8
class Normal(object):
    n_params = 2

    def __init__(self, params, temp_scale = 1.0):
        self.loc = params[0]
        self.scale = np.exp(params[1] / temp_scale) + 1e-8
        self.var = self.scale #** 2  + 1e-8
        self.shp = self.loc.shape

        self.dist = dist(loc=self.loc, scale=self.scale)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def nll(self, Y):
        return -self.dist.logpdf(Y)

    def D_nll(self, Y_):
        Y = Y_.squeeze()
        D = np.zeros((self.var.shape[0], 2))
        D[:, 0] = (self.loc - Y) / self.var
        D[:, 1] = 1 - ((self.loc - Y) ** 2) / self.var
        return D

    def crps(self, Y):
        Z = (Y - self.loc) / (self.scale + EPS)
        return (self.scale * (Z * (2 * sp.stats.norm.cdf(Z) - 1) +                 2 * sp.stats.norm.pdf(Z) - 1 / np.sqrt(np.pi)))

    def D_crps(self, Y_):
        Y = Y_.squeeze()
        Z = (Y - self.loc) / (self.scale + EPS)
        D = np.zeros((self.var.shape[0], 2))
        D[:, 0] = -(2 * sp.stats.norm.cdf(Z) - 1)
        D[:, 1] = self.crps(Y) + (Y - self.loc) * D[:, 0]
        return D

    def crps_metric(self):
        I = np.c_[2 * np.ones_like(self.var), np.zeros_like(self.var),
                  np.zeros_like(self.var), self.var]
        I = I.reshape((self.var.shape[0], 2, 2))
        I = 1/(2*np.sqrt(np.pi)) * I
        return I #+ 1e-4 * np.eye(2)

    def fisher_info(self):
        FI = np.zeros((self.var.shape[0], 2, 2))
        FI[:, 0, 0] = 1/self.var + 1e-5
        FI[:, 1, 1] = 2
        return FI

    def fisher_info_cens(self, T):
        nabla = np.array([self.pdf(T),
                          (T - self.loc) / self.scale * self.pdf(T)])
        return np.outer(nabla, nabla) / (self.cdf(T) * (1 - self.cdf(T))) + 1e-2 * np.eye(2)

    def fit(Y):
        m, s = sp.stats.norm.fit(Y)
        return np.array([m, np.log(s)])
        #return np.array([m, np.log(1e-5)])


class NGBoost(BaseEstimator):

    def __init__(self, Dist=Normal, Score=MLE(),
                 Base=default_tree_learner, natural_gradient=True,
                 n_estimators=500, learning_rate=0.01, minibatch_frac=1.0,
                 verbose=True, verbose_eval=100, tol=1e-4):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.init_params = None
        self.base_models = []
        self.scalings = []
        self.tol = tol

    def pred_param(self, X, max_iter=None):
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
        return params

    def sample(self, X, Y, params):
        if self.minibatch_frac == 1.0:
            return np.arange(len(Y)), X, Y, params
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np_rnd.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs,:], Y[idxs], params[idxs, :]

    def fit_base(self, X, grads):
        models = [self.Base().fit(X, g) for g in grads.T]
        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    def line_search(self, resids, start, Y, scale_init=1):
        S = self.Score
        D_init = self.Dist(start.T)
        loss_init = S.loss(D_init, Y)
        scale = scale_init
        while True:
            scaled_resids = resids * scale
            D = self.Dist((start - scaled_resids).T)
            loss = S.loss(D, Y)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if not np.isnan(loss) and (loss < loss_init or norm < self.tol) and               np.linalg.norm(scaled_resids, axis=1).mean() < 5.0:
                break
            scale = scale * 0.5
        self.scalings.append(scale)
        return scale

    def fit(self, X, Y, X_val = None, Y_val = None, train_loss_monitor = None, val_loss_monitor = None):

        loss_list = []
        val_loss_list = []
        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)
        if X_val is not None and Y_val is not None:
            val_params = self.pred_param(X_val)

        S = self.Score

        if not train_loss_monitor:
            train_loss_monitor = S.loss

        if not val_loss_monitor:
            val_loss_monitor = S.loss

        for itr in range(self.n_estimators):
            _, X_batch, Y_batch, P_batch = self.sample(X, Y, params)

            D = self.Dist(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch)]
            loss = loss_list[-1]
            grads = S.grad(D, Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads)
            scale = self.line_search(proj_grad, P_batch, Y_batch)

            params -= self.learning_rate * scale * np.array([m.predict(X) for m in self.base_models[-1]]).T

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= self.learning_rate * scale * np.array([m.predict(X_val) for m in self.base_models[-1]]).T
                val_loss = val_loss_monitor(self.Dist(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if len(val_loss_list) > 10 and np.mean(np.array(val_loss_list[-5:])) >                    np.mean(np.array(val_loss_list[-10:-5])):
                    if self.verbose:
                        print(f"== Quitting at iteration / VAL {itr} (val_loss={val_loss:.4f})")
                    break

            if self.verbose and int(self.verbose_eval) > 0 and itr % int(self.verbose_eval) == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                      f"norm={grad_norm:.4f}")

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        return self

    def fit_init_params_to_marginal(self, Y, iters=1000):
        try:
            E = Y['Event']
            T = Y['Time'].reshape((-1, 1))[E == 1]
        except:
            T = Y
        self.init_params = self.Dist.fit(T)
        return


    def pred_dist(self, X, max_iter=None):
        params = np.asarray(self.pred_param(X, max_iter))
        dist = self.Dist(params.T)
        return dist

    def predict(self, X):
        dist = self.pred_dist(X)
        return list(dist.loc.flatten())


# Feature extractor:

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.parser import parse

def clock_to_sec(s):
    min_str, sec_str, msec_str = s.split(':')
    return 60 * int(min_str) + int(sec_str)

def featurize(rows):
    featnames = []
    features = []
    for _, row in rows.iterrows():
        if row['NflId'] == row['NflIdRusher']:
            rusher = row['NflId']
            team = row['Team']
            
            featnames += ['Home']
            features += [int(team == 'home')]
            
            featnames += ['Clock']
            features += [clock_to_sec(row['GameClock'])]
            
            featnames += ['Quarter']
            features += [int(row['Quarter'])]
            
            if team == 'home':
                featnames += ['FriendScore']
                features += [int(row['HomeScoreBeforePlay'])]
                featnames += ['EnemyScore']
                features += [int(row['VisitorScoreBeforePlay'])]
            else:
                featnames += ['FriendScore']
                features += [int(row['VisitorScoreBeforePlay'])]
                featnames += ['EnemyScore']
                features += [int(row['HomeScoreBeforePlay'])]
                
            featnames += ['RusherHeight']
            ft, inch = row['PlayerHeight'].split('-')
            features += [int(ft) * 12 + int(inch)]
            
            featnames += ['RusherWeight']
            features += [int(row['PlayerWeight'])]
            
            featnames += ['RusherAge']
            features += [parse(row['TimeHandoff']).year - parse(row['PlayerBirthDate']).year]
            
            featnames += ['Week']
            features += [int(row['Week'])]
            
            featnames += ['Temperature']
            try:
                features += [int(row['Temperature'])]
            except:
                features += [60]
            
            featnames += ['Humidity']
            try:
                features += [int(row['Humidity'])]
            except:
                features += [55]
        
            # Orient the game rightwards always, and center on the ball/rusher
            right = row['PlayDirection'] == 'right'
            if right:
                ball_X0, ball_Y0, ball_S, ball_A, ball_D = row['X'], row['Y'], row['S'], row['A'], np.radians(row['Dir'])
            else:
                ball_X0, ball_Y0, ball_S, ball_A, ball_D = 120 - row['X'], 53.3 - row['Y'], row['S'], row['A'], np.radians((row['Dir'] + 180) % 360)
            break
            
            featnames += ['RusherSpeed']
            features += [ball_S]
    
            featnames += ['RusherDir']
            features += [ball_D]
            
            featnames += ['RusherAccel']
            features += [row['A']]
    
    friends, enemies = np.zeros((10, 5)), np.zeros((11, 5))
    fidx, eidx = 0, 0
    for _, row in rows.iterrows():
        if row['NflId'] == rusher:
            continue
        if right:
            vals = [row['X'], row['Y'], row['S'], row['A'], np.radians(row['Dir'])]
        else:
            vals = [120 - row['X'], 53.3 - row['Y'], row['S'], row['A'], np.radians((row['Dir'] + 180) % 360)]
        if team == row['Team']:
            friends[fidx, :] = vals
            fidx += 1
        else:
            enemies[eidx, :] = vals
            eidx += 1

    assert(fidx == 10)
    assert(eidx == 11)

    friends_rel_X0 = friends[:, 0] - ball_X0
    friends_rel_Y0 = friends[:, 1] - ball_Y0
    friends_R0 = (friends_rel_X0 ** 2 + friends_rel_Y0 ** 2) ** .5
    friends_Theta0 = np.arctan2(friends_rel_Y0, friends_rel_X0)

    enemies_rel_X0 = enemies[:, 0] - ball_X0
    enemies_rel_Y0 = enemies[:, 1] - ball_Y0
    enemies_R0 = (enemies_rel_X0 ** 2 + enemies_rel_Y0 ** 2) ** .5
    enemies_Theta0 = np.arctan2(enemies_rel_Y0, enemies_rel_X0)

    featnames += ['Ball_X0', 'Ball_Y0']
    features += [ball_X0, ball_Y0]

    Rs = [0, 3, 10, 30, 120]
    for r in range(len(Rs)-1):
        rl, rh = Rs[r], Rs[r+1]
        for t in range(8):
            tl, th = t * np.pi/8, (t+1) * np.pi/8
            key = 'map0_r%dq%d' % (rh, t)
            featnames.append('F' + key)
            features.append(np.sum(
                (rl <= friends_R0) & (friends_R0 < rh)
                & (tl <= friends_Theta0) & (friends_Theta0 < th)))
            featnames.append('E' + key)
            features.append(np.sum(
                (rl <= enemies_R0) & (enemies_R0 < rh)
                & (tl <= enemies_Theta0) & (enemies_Theta0 < th)))

    return featnames, features

def fparse(fname):
    df = pd.read_csv(fname, low_memory=False)

    Y = df[['PlayId', 'Yards']].groupby('PlayId').mean().values if 'Yards' in df.columns else None


    X = []
    for p in tqdm(range(int(df.shape[0] / 22))):
        rows = df.iloc[p * 22: p*22 + 22]
        names, features = featurize(rows)
        X.append(features)


    #X = df.groupby('PlayId').apply(grouper)

    return np.array(X), Y


# Load design matrix:

# In[ ]:


X, Y = fparse('/kaggle/input/nfl-big-data-bowl-2020/train.csv')


# Train model:

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
np.random.seed(123)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
ngb = NGBoost(Dist=Normal, Score=CRPS(), verbose=True, learning_rate=0.01, n_estimators=5000)
ngb.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val)

Y_dists = ngb.pred_dist(X_val)

# test Root Mean Squared Error
test_RMSE = np.sqrt(mean_squared_error(Y_dists.mean(), Y_val))
print('Val RMSE', test_RMSE)

# test CRPS
test_CRPS = Y_dists.crps(Y_val.flatten()).mean()
print('Val CRPS', test_CRPS)


# Make predictions:

# In[ ]:


from kaggle.competitions import nflrush

env = nflrush.make_env()
iter_test = env.iter_test()

Q = list(range(-99, 100))
for (batch, sample) in tqdm(iter_test):
    _, X_feats = featurize(batch)
    X_test = np.array([X_feats])
    Y_pred = ngb.pred_dist(X_test)
    sample.iloc[0] = Y_pred.cdf(Q)
    env.predict(sample)

env.write_submission_file()


#!/usr/bin/env python
# coding: utf-8

# ## Overview
# This kernel obtains an estimate of mean regional stellar magnitude changes in ~25 years for a dataset of 257K stars.
# 
# It uses magnitude change estimates from kernel *[Magnitude Change From Hipparcos To Gaia](https://www.kaggle.com/solorzano/magnitude-change-from-hipparcos-to-gaia)*. For each of 79K Hipparcos stars, the kernel gets a trimmed average of the estimated magnitude change of the star's K nearest neighbors. These results are then smoothed out with k-NN (k=5) on rectangular galactic coordinates.
# 
# The smoothing model is then applied to a bigger Gaia DR2 dataset, one with 257K stars. Because the Hipparcos dataset has a proximity bias, the results in the larger dataset evidently still carry this bias. This is a limitation of the results.
# 
# Regional magnitude change estimates are made available in the output tab of this kernel.
# 
# **Updates**:
# * 6/22/2019: Final smoothing is done with k-NN (k=5) instead of a Random Forest. This improves smoothing model performance.

# ## Modeling helper functions
# We will define our usual helper function that produces a transform function that takes a data frame and adds a model response column to it. Modeling relies on multiple runs of k-fold cross-validation, so all responses can be considered "out of bag." The transform function can be applied to data frames that were not used in training, so long as they contain all required variables.

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 
import types
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(2019050001)

def get_cv_model_transform(data_frame, label_extractor, var_extractor, trainer_factory, response_column='response', 
                           id_column='source_id', n_runs=2, n_splits=2, max_n_training=None, scale = False,
                           trim_fraction=None, classification=False):
    '''
    Creates a transform function that results from training a regression model with cross-validation.
    The transform function takes a frame and adds a response column to it.
    '''
    default_model_list = []
    sum_series = pd.Series([0] * len(data_frame))
    for r in range(n_runs):
        shuffled_frame = data_frame.sample(frac=1)
        shuffled_frame.reset_index(inplace=True, drop=True)
        response_frame = pd.DataFrame(columns=[id_column, 'response'])
        kf = KFold(n_splits=n_splits)
        first_fold = True
        for train_idx, test_idx in kf.split(shuffled_frame):
            train_frame = shuffled_frame.iloc[train_idx]
            if trim_fraction is not None:
                helper_labels = label_extractor(train_frame) if isinstance(label_extractor, types.FunctionType) else train_frame[label_extractor] 
                train_label_ordering = np.argsort(helper_labels)
                orig_train_len = len(train_label_ordering)
                head_tail_len_to_trim = int(round(orig_train_len * trim_fraction * 0.5))
                assert head_tail_len_to_trim > 0
                trimmed_ordering = train_label_ordering[head_tail_len_to_trim:-head_tail_len_to_trim]
                train_frame = train_frame.iloc[trimmed_ordering]
            if max_n_training is not None:
                train_frame = train_frame.sample(max_n_training)
            train_labels = label_extractor(train_frame) if isinstance(label_extractor, types.FunctionType) else train_frame[label_extractor]
            test_frame = shuffled_frame.iloc[test_idx]
            train_vars = var_extractor(train_frame)
            test_vars = var_extractor(test_frame)
            scaler = None
            if scale:
                scaler = StandardScaler()  
                scaler.fit(train_vars)
                train_vars = scaler.transform(train_vars)  
                test_vars = scaler.transform(test_vars) 
            trainer = trainer_factory()
            fold_model = trainer.fit(train_vars, train_labels)
            test_responses = fold_model.predict_proba(test_vars)[:,1] if classification else fold_model.predict(test_vars)
            test_id = test_frame[id_column]
            assert len(test_id) == len(test_responses)
            fold_frame = pd.DataFrame({id_column: test_id, 'response': test_responses})
            response_frame = pd.concat([response_frame, fold_frame], sort=False)
            if first_fold:
                first_fold = False
                default_model_list.append((scaler, fold_model,))
        response_frame.sort_values(id_column, inplace=True)
        response_frame.reset_index(inplace=True, drop=True)
        assert len(response_frame) == len(data_frame), 'len(response_frame)=%d' % len(response_frame)
        sum_series += response_frame['response']
    cv_response = sum_series / n_runs
    assert len(cv_response) == len(data_frame)
    assert len(default_model_list) == n_runs
    response_map = dict()
    sorted_id = np.sort(data_frame[id_column].values) 
    for i in range(len(cv_response)):
        response_map[str(sorted_id[i])] = cv_response[i]
    response_id_set = set(response_map)
    
    def _transform(_frame):
        _in_trained_set = _frame[id_column].astype(str).isin(response_id_set)
        _trained_frame = _frame[_in_trained_set].copy()
        _trained_frame.reset_index(inplace=True, drop=True)
        if len(_trained_frame) > 0:
            _trained_id = _trained_frame[id_column]
            _tn = len(_trained_id)
            _response = pd.Series([None] * _tn)
            for i in range(_tn):
                _response[i] = response_map[str(_trained_id[i])]
            _trained_frame[response_column] = _response
        _remain_frame = _frame[~_in_trained_set].copy()
        _remain_frame.reset_index(inplace=True, drop=True)
        if len(_remain_frame) > 0:
            _unscaled_vars = var_extractor(_remain_frame)
            _response_sum = pd.Series([0] * len(_remain_frame))
            for _model_tuple in default_model_list:
                _scaler = _model_tuple[0]
                _model = _model_tuple[1]
                _vars = _unscaled_vars if _scaler is None else _scaler.transform(_unscaled_vars)
                _response = _model.predict_proba(_vars)[:,1] if classification else _model.predict(_vars)
                _response_sum += _response
            _remain_frame[response_column] = _response_sum / len(default_model_list)
        _frames_list = [_trained_frame, _remain_frame]
        _result = pd.concat(_frames_list, sort=False)
        _result.reset_index(inplace=True, drop=True)
        return _result
    return _transform


# In[ ]:


import scipy.stats as stats

def print_evaluation(data_frame, label_column, response_column):
    response = response_column(data_frame) if isinstance(response_column, types.FunctionType) else data_frame[response_column]
    label = label_column(data_frame) if isinstance(label_column, types.FunctionType) else data_frame[label_column]
    residual = label - response
    rmse = np.sqrt(sum(residual ** 2) / len(data_frame))
    correl = stats.pearsonr(response, label)[0]
    print('RMSE: %.4f | Correlation: %.4f' % (rmse, correl,), flush=True)


# ## Magnitude change data
# Let's load data produced by kernel *[Magnitude Change From Hipparcos To Gaia](https://www.kaggle.com/solorzano/magnitude-change-from-hipparcos-to-gaia)*.

# In[ ]:


import pandas as pd

raw_data = pd.read_csv('../input/magnitude-change-from-hipparcos-to-gaia/hipparcos-gaia-mag-change-estimate.csv', dtype={'source_id': str})
len(raw_data)


# ## Regional magnitude change metric
# A region could have a few extreme outliers that completely change the apparent behavior of the region. Instead of calculating the average estimated change in magnitude, we use a 5% trimmed average. Trimmed averages are generally more robust estimators than averages. 

# In[ ]:


from scipy.stats import trim_mean

def regional_metric(x):
    return trim_mean(x, 0.05) 


# The training dataset is not too big. We will only look at 50 neighbors of each star.

# In[ ]:


K = 50
BASE_METRIC = 'mag_change_estimate'


# ## Limiting region bounds
# Ultimately we want to apply model results to a larger dataset that is bounded at 500 parsecs. Let's truncate the training dataset accordingly.

# In[ ]:


raw_data = raw_data[raw_data['parallax'] >= 2.0].reset_index(drop=True)


# In[ ]:


len(raw_data)


# ## Addition of rectangular coordinates
# We will add x-y-z coordinates to each row of the data frame. The *X* axis points to the galactic center.

# In[ ]:


import numpy as np

def get_position_frame(data_frame):
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame['source_id'] = data_frame['source_id'].values
    distance = None
    if 'distance' in data_frame.columns:
        distance = data_frame['distance'].values
    else:
        distance = 1000.0 / data_frame['parallax'].values
    latitude = np.deg2rad(data_frame['b'].values)
    longitude = np.deg2rad(data_frame['l'].values)
    new_frame['z'] = distance * np.sin(latitude)
    projection = distance * np.cos(latitude)
    new_frame['x'] = projection * np.cos(longitude)
    new_frame['y'] = projection * np.sin(longitude)
    if BASE_METRIC in data_frame.columns:
        new_frame[BASE_METRIC] = data_frame[BASE_METRIC]
    if 'anomaly' in data_frame.columns:
        new_frame['anomaly'] = data_frame['anomaly']
    if 'tycho2_id' in data_frame.columns:
        new_frame['tycho2_id'] = data_frame['tycho2_id']
    return new_frame


# In[ ]:


pos_data = get_position_frame(raw_data)
raw_data = None # Discard


# ## Main search routine
# In order to find a star's K nearest neighbors, we will use sklearn's BallTree implementation.

# In[ ]:


from sklearn.neighbors import BallTree

ball_tree = BallTree(pos_data[['x', 'y', 'z']])


# In[ ]:


metric_series = pos_data[BASE_METRIC].values
idx_source_id = pos_data.columns.get_loc('source_id')
idx_x = pos_data.columns.get_loc('x')
idx_y = pos_data.columns.get_loc('y')
idx_z = pos_data.columns.get_loc('z')
neighbors_mean_change = []
row_index = 0
for row in pos_data.itertuples(index=False):
    source_id = row[idx_source_id]
    source_pos = [row[idx_x], row[idx_y], row[idx_z]]
    # Get the K+1 nearest neighbors; results sorted by distance.
    distance_matrix, index_matrix = ball_tree.query([source_pos], k=K + 1)
    indexes = index_matrix[0]
    distances = distance_matrix[0]
    # The closest star is the current star - leave it out.
    assert indexes[0] == row_index
    assert distances[0] == 0
    leave_one_out_indexes = indexes[1:K + 1]
    metric_of_neighbors = metric_series[leave_one_out_indexes]    
    assert len(metric_of_neighbors) == K
    mc = regional_metric(metric_of_neighbors)
    neighbors_mean_change.append(mc)
    row_index += 1


# Column <code>raw_regional_metric</code> will contain the trimmed regional means calculated by looking at the K nearest neighbors of each star.

# In[ ]:


pos_data['raw_regional_metric'] = neighbors_mean_change


# In[ ]:


pos_data.head(5)


# ## Positional smoothing of regional magnitude change
# Regional means calculated so far are noisy. A Random Forest is a good choice to smooth them, but we find that k-NN with k=5 performs slightly better, and does not create rectangular boundary artifacts.

# In[ ]:


def get_pos_features(data_frame):
    return data_frame[['x', 'y', 'z']]


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

def get_pos_trainer():
    return KNeighborsRegressor(n_neighbors=5)


# Hyperparameters are chosen the way they normally are in any machine learning optimization exercise.

# In[ ]:


pos_transform = get_cv_model_transform(pos_data, 'raw_regional_metric', get_pos_features, get_pos_trainer, 
    response_column='smooth_regional_metric', n_runs=3, n_splits=3, max_n_training=None, scale=True)


# In[ ]:


pos_data = pos_transform(pos_data)


# In[ ]:


print_evaluation(pos_data, 'raw_regional_metric', 'smooth_regional_metric')


# ## Apply model to bigger dataset
# We will load a bigger Gaia DR2 dataset: The one produced by kernel <i>[New Stellar Magnitude Model (Dysonian SETI)](https://www.kaggle.com/solorzano/new-stellar-magnitude-model-dysonian-seti)</i>. And we will add rectangular coordinates to it.

# In[ ]:


import pandas as pd

main_data = get_position_frame(pd.read_csv('../input/new-stellar-magnitude-model-dysonian-seti/mag-modeling-results.csv', dtype={'source_id': str}))


# Now we apply the Random Forest model transform to this data frame, which should add a <code>smooth_regional_metric</code> column to it.

# In[ ]:


main_data = pos_transform(main_data)


# ## Distribution of smooth regional magnitude change metric
# Let's look at a histogram of the modeled/smooth regional magnitude change metric. The histogram will be scaled so we can more clearly observe the tails of the distribution.

# In[ ]:


import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
py.init_notebook_mode(connected=False)


# In[ ]:


def plot_distribution(x, xlabel, xrange=None, yrange=None, xbins=None, lineX=None):
    hist_trace = go.Histogram(
        x=x,
        xbins=xbins,
    )
    data = [hist_trace]
    shapes = None
    if lineX is not None:
        shapes = [
            { 
                'type': 'line', 'x0': lineX, 'x1': lineX, 'y0': 0, 'y1': len(x), 
                'line': { 'color': 'orange', 'width': 3, 'dash': 'dashdot',},
            }
        ]
    layout = go.Layout(barmode='overlay', 
        xaxis=dict(
            title=xlabel,
            range=xrange,
        ),
        yaxis=dict(
            title='Frequency',
            range=yrange,
        ),
        shapes=shapes,
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


# In[ ]:


SELECTED_THRESHOLD = 0.012

plot_distribution(main_data['smooth_regional_metric'], 
        xlabel='Smooth magnitude change mean', 
        yrange=[0, 100],
        lineX = SELECTED_THRESHOLD)


# The distribution is visibly right-tailed. Positive magnitude changes indicate dimming.

# ## Visualization of regions with anomalous dimming
# We will select stars whose modeled regional metric is to the right of the dashed line in the histogram, for visualization purposes.

# In[ ]:


high_outliers = main_data[main_data['smooth_regional_metric'] >= SELECTED_THRESHOLD]


# In[ ]:


len(high_outliers)


# Now we include our usual boilerplate code for rendering stars in interactive 3D scatter charts.

# In[ ]:


def plot_pos_frame(pos_frame, star_color, sun_color = 'blue', bstar_color = 'black'):    
    star_color = [(bstar_color if row['source_id'] == '2081900940499099136' else (sun_color if row['source_id'] == 'sun' else star_color)) for _, row in pos_frame.iterrows()]
    trace1 = go.Scatter3d(
        x=pos_frame['x'],
        y=pos_frame['y'],
        z=pos_frame['z'],
        mode='markers',
        text=pos_frame['source_id'],
        marker=dict(
            size=3,
            color=star_color,
            opacity=0.5
        )
    )
    scatter_data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            xaxis=dict(
                range=[-450, 450]
            ),
            yaxis=dict(
                range=[-450, 450]
            ),
            zaxis=dict(
                range=[-450, 450]
            )
        ),
    )
    fig = go.Figure(data=scatter_data, layout=layout)
    py.iplot(fig)


# The sun will be rendered in blue.

# In[ ]:


def get_sun():
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame.loc[0] = ['sun', 0.0, 0.0, 0.0]
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame


# We will also render KIC 8462852, for reference, in black.

# In[ ]:


bstar_frame = main_data[main_data['source_id'] == '2081900940499099136']
high_outliers_with_extras = pd.concat([high_outliers, bstar_frame, get_sun()])


# In[ ]:


plot_pos_frame(high_outliers_with_extras, 'green')


# The results seem to be biased toward proximity to the sun. This is unsurprising given the nature of the Hipparcos dataset that was used to derive individual magnitude change estimates.

# ## Magnitude change distributions
# Is the distribution of Hipparcos-Gaia magnitude change estimates different in the regions with a high average dimming trend compared to other regions? Let's look at a couple of histograms. We will take the top 1000 stars in Hipparcos according to the smooth regional average, and will compare them to 1000 stars taken at random from ordinary regions.

# In[ ]:


N_HIST = 1000
hip_outliers = pos_data.sort_values('smooth_regional_metric', ascending=False).head(N_HIST)


# In[ ]:


outside_region = pos_data[~pos_data['source_id'].isin(set(hip_outliers['source_id']))].sample(len(hip_outliers))


# In[ ]:


plot_distribution(hip_outliers[BASE_METRIC], xlabel='Hipparcos-Gaia Magnitude Change in Anomalous Regions', xrange=[-0.3, +1.0], yrange=[0, 100])


# In[ ]:


plot_distribution(outside_region[BASE_METRIC], xlabel='Hipparcos-Gaia Magnitude Change in Ordinary Regions', xrange=[-0.3, +1.0], yrange=[0, 100])


# ## Writing output files
# Let's write the <code>main_data</code> frame, which includes rectangular positions and the new <code>smooth_regional_metric</code> column.

# In[ ]:


main_data[['source_id', 'smooth_regional_metric']].to_csv('region-hip-gaia-mean-mag-change.csv', index=False)


# For reference, we'll also save the data that was used to train the smoothing model.

# In[ ]:


pos_data[['source_id', 'x', 'y', 'z', 'raw_regional_metric', 'smooth_regional_metric']].to_csv('mean-mag-change-training-data.csv', index=False)


# ## Acknowledgements
# 
# This work has made use of data from the European Space Agency (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the Gaia Multilateral Agreement.
# 
# The Hipparcos and Tycho Catalogues are the primary products of the European Space Agency's astrometric mission, Hipparcos. The satellite, which operated for four years, returned high quality scientific data from November 1989 to March 1993.

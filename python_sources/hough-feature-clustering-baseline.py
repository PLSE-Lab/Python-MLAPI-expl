# Altered from 
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
#import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN



"""TrackML dataset loading"""

__authors__ = ['Moritz Kiehn', 'Sabrina Amrouche']

import glob
import os
import os.path as op
import re
import zipfile

import pandas

CELLS_DTYPES = dict([
    ('hit_id', 'i4'),
    ('ch0', 'i4'),
    ('ch1', 'i4'),
    ('value', 'f4'),
])
HITS_DTYPES = dict([
    ('hit_id', 'i4'),
    ('x', 'f4'),
    ('y', 'f4'),
    ('z','f4'),
    ('volume_id', 'i4'),
    ('layer_id', 'i4'),
    ('module_id', 'i4'),
])
PARTICLES_DTYPES = dict([
    ('particle_id', 'i8'),
    ('vx', 'f4'),
    ('vy', 'f4'),
    ('vz', 'f4'),
    ('px', 'f4'),
    ('py', 'f4'),
    ('pz', 'f4'),
    ('q', 'i4'),
    ('nhits', 'i4'),
])
TRUTH_DTYPES = dict([
    ('hit_id', 'i4'),
    ('particle_id', 'i8'),
    ('tx', 'f4'),
    ('ty', 'f4'),
    ('tz', 'f4'),
    ('tpx', 'f4'),
    ('tpy', 'f4'),
    ('tpz', 'f4'),
    ('weight', 'f4'),
])
DTYPES = {
    'cells': CELLS_DTYPES,
    'hits': HITS_DTYPES,
    'particles': PARTICLES_DTYPES,
    'truth': TRUTH_DTYPES,
}
DEFAULT_PARTS = ['hits', 'cells', 'particles', 'truth']

def _load_event_data(prefix, name):
    """Load per-event data for one single type, e.g. hits, or particles.
    """
    expr = '{!s}-{}.csv*'.format(prefix, name)
    files = glob.glob(expr)
    dtype = DTYPES[name]
    if len(files) == 1:
        return pandas.read_csv(files[0], header=0, index_col=False, dtype=dtype)
    elif len(files) == 0:
        raise Exception('No file matches \'{}\''.format(expr))
    else:
        raise Exception('More than one file matches \'{}\''.format(expr))

def load_event_hits(prefix):
    """Load the hits information for a single event with the given prefix.
    """
    return _load_event_data(prefix, 'hits')

def load_event_cells(prefix):
    """Load the hit cells information for a single event with the given prefix.
    """
    return _load_event_data(prefix, 'cells')

def load_event_particles(prefix):
    """Load the particles information for a single event with the given prefix.
    """
    return _load_event_data(prefix, 'particles')

def load_event_truth(prefix):
    """Load only the truth information for a single event with the given prefix.
    """
    return _load_event_data(prefix, 'truth')

def load_event(prefix, parts=DEFAULT_PARTS):
    """Load data for a single event with the given prefix.
    Parameters
    ----------
    prefix : str or pathlib.Path
        The common prefix name for the event files, i.e. without `-hits.csv`).
    parts : List[{'hits', 'cells', 'particles', 'truth'}], optional
        Which parts of the event files to load.
    Returns
    -------
    tuple
        Contains a `pandas.DataFrame` for each element of `parts`. Each
        element has field names identical to the CSV column names with
        appropriate types.
    """
    return tuple(_load_event_data(prefix, name) for name in parts)

def load_dataset(path, skip=None, nevents=None, parts=DEFAULT_PARTS):
    """Provide an iterator over (all) events in a dataset.
    Parameters
    ----------
    path : str or pathlib.Path
        Path to a directory or a zip file containing event files.
    skip : int, optional
        Skip the first `skip` events.
    nevents : int, optional
        Only load a maximum of `nevents` events.
    parts : List[{'hits', 'cells', 'particles', 'truth'}], optional
        Which parts of each event files to load.
    Yields
    ------
    event_id : int
        The event identifier.
    *data
        Event data element as specified in `parts`.
    """
    # extract a sorted list of event file prefixes.
    def list_prefixes(files):
        regex = re.compile('^event\d{9}-[a-zA-Z]+.csv')
        files = filter(regex.match, files)
        prefixes = set(op.basename(_).split('-', 1)[0] for _ in files)
        prefixes = sorted(prefixes)
        if skip is not None:
            prefixes = prefixes[skip:]
        if nevents is not None:
            prefixes = prefixes[:nevents]
        return prefixes

    # TODO use yield from when we increase the python requirement
    if op.isdir(path):
        for x in _iter_dataset_dir(path, list_prefixes(os.listdir(path)), parts):
            yield x
    else:
        with zipfile.ZipFile(path, mode='r') as z:
            for x in _iter_dataset_zip(z, list_prefixes(z.namelist()), parts):
                yield x

def _extract_event_id(prefix):
    """Extract event_id from prefix, e.g. event_id=1 from `event000000001`.
    """
    return int(prefix[5:])

def _iter_dataset_dir(directory, prefixes, parts):
    """Iterate over selected events files inside a directory.
    """
    for p in prefixes:
        yield (_extract_event_id(p),) + load_event(op.join(directory, p), parts)

def _iter_dataset_zip(zipfile, prefixes, parts):
    """"Iterate over selected event files inside a zip archive.
    """
    for p in prefixes:
        files = [zipfile.open('{}-{}.csv'.format(p, _), mode='r') for _ in parts]
        dtypes = [DTYPES[_] for _ in parts]
        data = tuple(pandas.read_csv(f, header=0, index_col=False, dtype=d)
                                     for f, d in zip(files, dtypes))
        yield (_extract_event_id(p),) + data
        
"""TrackML scoring metric"""

__authors__ = ['Sabrina Amrouche', 'David Rousseau', 'Moritz Kiehn',
               'Ilija Vukotic']

import numpy
import pandas

def _analyze_tracks(truth, submission):
    """Compute the majority particle, hit counts, and weight for each track.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
    """
    # true number of hits for each particle_id
    particles_nhits = truth['particle_id'].value_counts(sort=False)
    total_weight = truth['weight'].sum()
    # combined event with minimal reconstructed and truth information
    event = pandas.merge(truth[['hit_id', 'particle_id', 'weight']],
                         submission[['hit_id', 'track_id']],
                         on=['hit_id'], how='left', validate='one_to_one')
    event.drop('hit_id', axis=1, inplace=True)
    event.sort_values(by=['track_id', 'particle_id'], inplace=True)

    # ASSUMPTIONs: 0 <= track_id, 0 <= particle_id

    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    cur_weight = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0
    maj_weight = 0

    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                particles_nhits[maj_particle_id], maj_nhits,
                maj_weight / total_weight))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
            maj_particle_id = -1
            maj_nhits = 0
            maj_weights = 0
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1
            cur_weight += hit.weight

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits
        maj_weight = cur_weight
    # store values for the last track
    tracks.append((rec_track_id, rec_nhits, maj_particle_id,
        particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

    cols = ['track_id', 'nhits',
            'major_particle_id', 'major_particle_nhits',
            'major_nhits', 'major_weight']
    return pandas.DataFrame.from_records(tracks, columns=cols)

def score_event(truth, submission):
    """Compute the TrackML event score for a single event.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    """
    tracks = _analyze_tracks(truth, submission)
    purity_rec = numpy.divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = numpy.divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track_idx = (0.5 < purity_rec) & (0.5 < purity_maj)
	#good_track_df = tracks[good_track_idx]
    return tracks['major_weight'][good_track_idx].sum()
	
	
def score_event_kha(truth, submission):
    """Compute the TrackML event score for a single event.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    """

    tracks = _analyze_tracks(truth, submission)
    purity_rec = numpy.divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = numpy.divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
	#sensentivity = 
    return tracks['major_weight'][good_track].sum()


# In[3]:


def merge(cl1, cl2): # merge cluster 2 to cluster 1
    d = pd.DataFrame(data={'s1':cl1,'s2':cl2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    maxs1 = d['s1'].max()
    cond = np.where((d['N2'].values>d['N1'].values) & (d['N2'].values<25)) # Locate the hit with the new cluster> old cluster
    s1 = d['s1'].values 
    s1[cond] = d['s2'].values[cond]+maxs1 # Assign all hits that belong to the new track (+ maxs1 to increase the label for the track so it's different from the original).
    return s1

def extract_good_hits(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    return tru[(tru.count_both > 0.5*tru.count_particle) & (tru.count_both > 0.5*tru.count_track)]

def fast_score(good_hits_df):
    return good_hits_df.weight.sum()


def analyze_truth_perspective(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    good_hits = tru[(tru.count_both > 0.5*tru.count_particle) & (tru.count_both > 0.5*tru.count_track)]
    score = good_hits.weight.sum()
    
    anatru = tru.particle_id.value_counts().value_counts().sort_index().to_frame().rename({'particle_id':'true_particle_counts'}, axis=1)
    #anatru['true_particle_ratio'] = anatru['true_particle_counts'].values*100/np.sum(anatru['true_particle_counts'])

    anatru['good_tracks_counts'] = np.zeros(len(anatru)).astype(int)
    anatru['good_tracks_intersect_nhits_avg'] = np.zeros(len(anatru))
    anatru['best_detect_intersect_nhits_avg'] = np.zeros(len(anatru))
    for nhit in tqdm(range(4,20)):
        particle_list  = tru[(tru.count_particle==nhit)].particle_id.unique()
        intersect_count = 0
        good_tracks_count = 0
        good_tracks_intersect = 0
        for p in particle_list:
            nhit_intersect = tru[tru.particle_id==p].count_both.max()
            intersect_count += nhit_intersect
            corresponding_track = tru.loc[tru[tru.particle_id==p].count_both.idxmax()].track_id
            leng_corresponding_track = len(tru[tru.track_id == corresponding_track])
            
            if (nhit_intersect >= nhit/2) and (nhit_intersect >= leng_corresponding_track/2):
                good_tracks_count += 1
                good_tracks_intersect += nhit_intersect
        intersect_count = intersect_count/len(particle_list)
        anatru.at[nhit,'best_detect_intersect_nhits_avg'] = intersect_count
        anatru.at[nhit,'good_tracks_counts'] = good_tracks_count
        if good_tracks_count > 0:
            anatru.at[nhit,'good_tracks_intersect_nhits_avg'] = good_tracks_intersect/good_tracks_count
    
    return score, anatru, good_hits

def precision(truth, submission,min_hits):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    #print('Analyzing predictions...')
    predicted_list  = tru[(tru.count_track>=min_hits)].track_id.unique()
    good_tracks_count = 0
    ghost_tracks_count = 0
    fp_weights = 0
    tp_weights = 0
    for t in predicted_list:
        nhit_track = tru[tru.track_id==t].count_track.iloc[0]
        nhit_intersect = tru[tru.track_id==t].count_both.max()
        corresponding_particle = tru.loc[tru[tru.track_id==t].count_both.idxmax()].particle_id
        leng_corresponding_particle = len(tru[tru.particle_id == corresponding_particle])
        if (nhit_intersect >= nhit_track/2) and (nhit_intersect >= leng_corresponding_particle/2): #if the predicted track is good
            good_tracks_count += 1
            tp_weights += tru[(tru.track_id==t)&(tru.particle_id==corresponding_particle)].weight.sum()
            fp_weights += tru[(tru.track_id==t)&(tru.particle_id!=corresponding_particle)].weight.sum()
        else: # if the predicted track is bad
                ghost_tracks_count += 1
                fp_weights += tru[(tru.track_id==t)].weight.sum()
    all_weights = tru[(tru.count_track>=min_hits)].weight.sum()
    precision = tp_weights/all_weights*100
    print('Precision: ',precision,', good tracks:', good_tracks_count,', total tracks:',len(predicted_list),
           ', loss:', fp_weights, ', reco:', tp_weights, 'reco/loss', tp_weights/fp_weights)
    return precision


class Clusterer(object):
    def __init__(self):                        
        self.abc = []
          
    def initialize(self,dfhits):
        self.cluster = range(len(dfhits))
        
    def Hough_clustering(self,dfh,coef,epsilon,min_samples=1,n_loop=180,verbose=True): # [phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        merged_cluster = self.cluster
        mm = 1
        stepii = 0.000005
        count_ii = 0
        adaptive_eps_coefficient = 1
        for ii in np.arange(0, n_loop*stepii, stepii):
            count_ii += 1
            for jj in range(2):
                mm = mm*(-1)
                eps_new = epsilon + count_ii*adaptive_eps_coefficient*10**(-5)
                dfh['a1'] = dfh['a0'].values - np.nan_to_num(np.arccos(mm*ii*dfh['rt'].values))
                dfh['sina1'] = np.sin(dfh['a1'].values)
                dfh['cosa1'] = np.cos(dfh['a1'].values)
                ss = StandardScaler()
                dfs = ss.fit_transform(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']].values) 
                #dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
                dfs = np.multiply(dfs, coef)
                new_cluster=DBSCAN(eps=eps_new,min_samples=min_samples,metric='euclidean',n_jobs=4).fit(dfs).labels_
                merged_cluster = merge(merged_cluster, new_cluster)
                if verbose == True:
                    sub = create_one_event_submission(0, hits, merged_cluster)
                    good_hits = extract_good_hits(truth, sub)
                    score_1 = fast_score(good_hits)
                    print('2r0_inverse:', ii*mm ,'. Score:', score_1)
                    #clear_output(wait=True)
        self.cluster = merged_cluster

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission  

def preprocess_hits(h,dz):
    h['z'] =  h['z'].values + dz
    h['r'] = np.sqrt(h['x'].values**2+h['y'].values**2+h['z'].values**2)
    h['rt'] = np.sqrt(h['x'].values**2+h['y'].values**2)
    h['a0'] = np.arctan2(h['y'].values,h['x'].values)
    h['zdivrt'] = h['z'].values/h['rt'].values
    h['zdivr'] = h['z'].values/h['r'].values
    h['xdivr'] = h['x'].values / h['r'].values
    h['ydivr'] = h['y'].values / h['r'].values
    return h

for i in tqdm(range(125)):
    path_to_train ="../input/test"
    event_prefix = "event"+str(i).zfill(9)
    hits = load_event_hits(os.path.join(path_to_train, event_prefix))
    c = [1.5,1.5,0.73,0.17,0.027,0.027] #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
    min_samples_in_cluster = 1

    model = Clusterer()
    model.initialize(hits) 
    hits_with_dz = preprocess_hits(hits, 0)
    model.Hough_clustering(hits_with_dz,coef=c,epsilon=0.0048,min_samples=min_samples_in_cluster,
                           n_loop=250,verbose=False)

    if i == 0:
        submission = create_one_event_submission(i, hits, model.cluster)
    else:
        submission = pd.concat([submission,create_one_event_submission(i, hits, model.cluster)])
   # print(submission)
   # submission.to_csv('submission.csv')
print('\n') 
submission.to_csv('submission.csv')

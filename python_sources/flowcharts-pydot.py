#!/usr/bin/env python
# coding: utf-8

# ## Overview
# Various code to make flowcharts and diagrams using python since Visio and powerpoint are too annoying

# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import StringIO
from IPython.display import SVG
import pydot # import pydot or you're not going to get anywhere my friend :D


# # Data Warehouse / Lake Strategy

# In[ ]:


dot_graph = pydot.Dot(graph_type='digraph')

sd_node = pydot.Node('Structured\nData\n(ODBC, CSV, XLS)')
sd_node.set_shape('box3d')
dot_graph.add_node(sd_node)

qsd_node = pydot.Node('Quasistructured\nData\n(PACS, RIS, JSON)')
qsd_node.set_shape('box3d')
dot_graph.add_node(qsd_node)

sds_node = pydot.Node('Streaming\nData Sources\n(0MQ, Kafka, IoT)')
sds_node.set_shape('box3d')
dot_graph.add_node(sds_node)

usd_node = pydot.Node('Unstructured\nData\n(Text, DOCX)')
usd_node.set_shape('box3d')
dot_graph.add_node(usd_node)

nlp_node = pydot.Node('NLP Processing\n(Word2Vec,\nBag of Words,\nParseNet)')
nlp_node.set_shape('box')
dot_graph.add_node(nlp_node)

riq_node = pydot.Node('4Quant\nAnalytics\nEngine')
#riq_node.set_shape('box3d')
dot_graph.add_node(riq_node)


iedge = pydot.Edge(sd_node,riq_node)
iedge.set_label('Tables')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(qsd_node,riq_node)
iedge.set_label('Key-Value\nStores')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(usd_node,nlp_node)
iedge.set_label('Keyword\nExtraction')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(usd_node,nlp_node)
iedge.set_label('Category')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(nlp_node,riq_node)
iedge.set_label('Key-Value\nStore')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(sds_node,riq_node)
iedge.set_label('Minibatch\nDatasets')
iedge.set_style('dashed')
dot_graph.add_edge(iedge)


asp_node = pydot.Node('Apache Spark')
asp_node.set_shape('square')
dot_graph.add_node(asp_node)

hadoop_node = pydot.Node('Distributed\nHadoop\nFilesystem (HDFS)')
hadoop_node.set_shape('box3d')
dot_graph.add_node(hadoop_node)

iedge = pydot.Edge(riq_node,asp_node)
iedge.set_label('Redundant\nDistributed\nDatasets')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(riq_node,asp_node)
iedge.set_label('Streaming\nReduntant\nDistributed\nDatasets')
iedge.set_style('dashed')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(asp_node,hadoop_node)
iedge.set_label('Parquet\nDistributed\nColumn\nStore')
iedge.set_penwidth(3)
dot_graph.add_edge(iedge)

odbc_node = pydot.Node('OBDC-Hive\nData View')
odbc_node.set_shape('box3d')
dot_graph.add_node(odbc_node)

xls_node = pydot.Node('Excel Workbook\nSummaries')
xls_node.set_shape('box')
dot_graph.add_node(xls_node)

iedge = pydot.Edge(asp_node,odbc_node)
iedge.set_penwidth(2)
dot_graph.add_edge(iedge)

iedge = pydot.Edge(odbc_node,xls_node)
iedge.set_penwidth(2)
dot_graph.add_edge(iedge)

sap_node = pydot.Node('SAP HANA')
sap_node.set_shape('none')
dot_graph.add_node(sap_node)

iedge = pydot.Edge(odbc_node,sap_node)
iedge.set_penwidth(1)
dot_graph.add_edge(iedge)


ibot_node = pydot.Node('Image\nBot')
ibot_node.set_shape('triangle')
dot_graph.add_node(ibot_node)


iedge = pydot.Edge(riq_node,ibot_node)
iedge.set_penwidth(1)
iedge.set_style('dashed')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(ibot_node, riq_node)
iedge.set_penwidth(1)
iedge.set_style('dashed')
dot_graph.add_edge(iedge)

ibot_node = pydot.Node('Genomics\nBot')
ibot_node.set_shape('triangle')
dot_graph.add_node(ibot_node)


iedge = pydot.Edge(riq_node,ibot_node)
iedge.set_penwidth(1)
iedge.set_style('dashed')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(ibot_node, riq_node)
iedge.set_penwidth(1)
iedge.set_style('dashed')
dot_graph.add_edge(iedge)

ibot_node = pydot.Node('DxRx\nBot')
ibot_node.set_shape('triangle')
dot_graph.add_node(ibot_node)


iedge = pydot.Edge(riq_node,ibot_node)
iedge.set_penwidth(1)
iedge.set_style('dashed')
dot_graph.add_edge(iedge)

iedge = pydot.Edge(ibot_node, riq_node)
iedge.set_penwidth(1)
iedge.set_style('dashed')
dot_graph.add_edge(iedge)


dot_graph.write_svg('big_data.svg')
dot_graph.write_ps2('big_data.ps2')
SVG('big_data.svg')


# # LungStage Workflow
# 

# In[ ]:


dot_graph = pydot.Dot(graph_type='digraph')

def make_node(name,shape):
    cur_node = pydot.Node(name)
    cur_node.set_shape(shape)
    dot_graph.add_node(cur_node)
    return cur_node

def make_link(a_node, b_node, label = None, width = 1, style='dashed'):
    cur_edge = pydot.Edge(a_node,b_node)
    cur_edge.set_penwidth(width)
    cur_edge.set_style(style)
    if label is not None: cur_edge.set_label(label)
    dot_graph.add_edge(cur_edge)
    return cur_edge

sd_node = make_node('Patient\n(Suspicious)','tab')
ct_node = make_node('Chest CT','box3d')
pet_node = make_node('PET/CT','box3d')
gp_node = make_node('General\nPractioner','box')
staging_node = make_node('Staging\n(Radiologist &\nNuclear Medicine)','box')
brocho_node = make_node('Bronchoscopy /\n Mediastinoscopy/ \nPathology','box')
diag_node = make_node('Diagnosis','trapezium')
tb_node = make_node('Tumor Board','component')

therapy_node = make_node('Therapy','cds')


make_link(gp_node,ct_node, 'Initial\nAssessment', style='solid')
make_link(sd_node,gp_node,'Coughing\nBlood', style='solid')

make_link(ct_node,gp_node,'Nothing\nFound')
make_link(ct_node,pet_node,'Suspicious\nFindings', style='solid')

make_link(pet_node,staging_node, style='solid')

make_link(staging_node, brocho_node, style='solid')
make_link(staging_node,gp_node,'Nothing\nFound')

make_link(brocho_node, diag_node, style='solid')
st_link = make_link(staging_node, tb_node, 'Prepare\nReport', style='thick')
st_link.set_color('blue')
st_link.set_fontcolor('blue')
st_link.set_fontsize(20)
st_link.set_penwidth(2.0)
st_link.set_style('dashed')
make_link(diag_node, tb_node, style='solid')


make_link(tb_node, therapy_node, 'Thorasic Surgery\n(Surgery)', style='solid')
make_link(tb_node, therapy_node, 'Chemotherapy\n(Oncology)', style='solid')
make_link(tb_node, therapy_node, 'Pallative\nCare\n(Oncology)')

make_link(therapy_node, ct_node, 'Follow-up', style='solid')
make_link(therapy_node, pet_node, 'Follow-up', style='solid')

dot_graph.set_overlap(False)
dot_graph.set_rankdir('UD')
dot_graph.write_svg('nsclc.svg', prog = 'dot')
dot_graph.write_ps2('nsclc.ps2')
SVG('nsclc.svg')


# In[ ]:


dot_graph = pydot.Dot(graph_type='digraph')

tb_node = make_node('Tumor Board','component')
therapy_node = make_node('Therapy','cds')


surg_node = make_node('Surgery','octagon')
onco_node = make_node('Oncology','octagon')

ct_node = make_node('Chest CT','box3d')
pet_node = make_node('PET/CT','box3d')
staging_node = make_node('Staging\n(Radiologist &\nNuclear Medicine)','box')

make_link(tb_node, therapy_node, 'Treatment', style='solid')
make_link(therapy_node, surg_node, 'Thorasic Surgery', style='solid')
make_link(therapy_node, onco_node, 'Chemotherapy', style='solid')
make_link(therapy_node, onco_node, 'Pallative\nCare', style='solid')
make_link(onco_node, surg_node, 'Thorasic Surgery')

make_link(onco_node, ct_node, 'Follow-up', style = 'solid')
make_link(surg_node, ct_node, 'Follow-up', style = 'solid')
make_link(ct_node, pet_node, 'More Detailed\nImages')
make_link(ct_node, staging_node, style = 'solid')
make_link(pet_node, staging_node, style = 'solid')
make_link(staging_node, tb_node, style = 'solid')


dot_graph.write_svg('nsclc_therapy.svg')
dot_graph.write_ps2('nsclc_therapy.ps2')
SVG('nsclc_therapy.svg')


# # Data Flow LungStage

# In[ ]:


dot_graph = pydot.Dot(graph_type='digraph')

def make_node(name,shape):
    cur_node = pydot.Node(name)
    cur_node.set_shape(shape)
    dot_graph.add_node(cur_node)
    return cur_node

def make_link(a_node, b_node, label = None, width = 1, style='dashed'):
    cur_edge = pydot.Edge(a_node,b_node)
    cur_edge.set_penwidth(width)
    cur_edge.set_style(style)
    if label is not None: cur_edge.set_label(label)
    dot_graph.add_edge(cur_edge)
    return cur_edge

nm_data_node_xml = make_node('Nuclear Medicine Records (XML)','tab')
nm_data_node_xls = make_node('Nuclear Medicine Records (XLS)','tab')
nm_data_node_adb = make_node('Nuclear Medicine Records (AccessDB)','tab')
nm_data_node = make_node('Aggregate NM Data','trapezium')
make_link(nm_data_node_xml, nm_data_node)
make_link(nm_data_node_xls, nm_data_node)
make_link(nm_data_node_adb, nm_data_node)

onco_data_node = make_node('Oncology Records (XLS)','tab')
join_node = make_node('Merge Records','trapezium')
make_link(onco_data_node, join_node)
make_link(nm_data_node, join_node, 'Filter on disease\ncase and type')

pacs_node = make_node('PACS Records','cds')
solr_node = make_node('PACSCrawler/SOLR Database','box')
pacs_join_node = make_node('Match PET Scans','trapezium')
make_link(pacs_node, solr_node, 'pypacscrawler\nmeta')
make_link(solr_node, pacs_join_node, '')
make_link(join_node, pacs_join_node, '')

master_list_node = make_node('Save as full_list.json','tab')
make_link(pacs_join_node, master_list_node, 'Master Patient List')

screened_list_node = make_node('Screened List','box')
make_link(pacs_join_node, screened_list_node, 'LungStage Screener\nGoogle Docs\n(Gregor)\n5min/patient')

download_node = make_node('Download for Annotation','box3d')
make_link(screened_list_node, download_node, '')

pet_download_node = make_node('Region Annotations','box3d')
make_link(download_node, pet_download_node, 'LungStage Annotation\nSlicer-based Tool\n(Alex and Thomas)\n30min/patient')
make_link(pet_download_node, screened_list_node, 'Filter already\nannotated cases')
region_list_node = make_node('Region Data\nlsa.npz\nlsa.json','tab')
make_link(pet_download_node, region_list_node, 'Save Region Data')

onco_list_node = make_node('Oncology List','box')
make_link(region_list_node, onco_list_node, '')
make_link(master_list_node, onco_list_node, '')

full_list_node = make_node('Full Patient Data','box')
make_link(onco_list_node, full_list_node, 'LungStage Oncology Tool\nGoogle Docs\n(Audrey)\n30min/patient')

if False:
    sd_node = make_node('Patient\n(Suspicious)','tab')
    ct_node = make_node('Chest CT','box3d')
    pet_node = make_node('PET/CT','box3d')
    gp_node = make_node('General\nPractioner','box')
    staging_node = make_node('Staging\n(Radiologist &\nNuclear Medicine)','box')
    brocho_node = make_node('Bronchoscopy /\n Mediastinoscopy/ \nPathology','box')
    diag_node = make_node('Diagnosis','trapezium')
    tb_node = make_node('Tumor Board','component')

    therapy_node = make_node('Therapy','cds')


    make_link(gp_node,ct_node, 'Initial\nAssessment', style='solid')
    make_link(sd_node,gp_node,'Coughing\nBlood', style='solid')

    make_link(ct_node,gp_node,'Nothing\nFound')
    make_link(ct_node,pet_node,'Suspicious\nFindings', style='solid')

    make_link(pet_node,staging_node, style='solid')

    make_link(staging_node, brocho_node, style='solid')
    make_link(staging_node,gp_node,'Nothing\nFound')

    make_link(brocho_node, diag_node, style='solid')
    st_link = make_link(staging_node, tb_node, 'Prepare\nReport', style='thick')
    st_link.set_color('blue')
    st_link.set_fontcolor('blue')
    st_link.set_fontsize(20)
    st_link.set_penwidth(2.0)
    st_link.set_style('dashed')
    make_link(diag_node, tb_node, style='solid')


    make_link(tb_node, therapy_node, 'Thorasic Surgery\n(Surgery)', style='solid')
    make_link(tb_node, therapy_node, 'Chemotherapy\n(Oncology)', style='solid')
    make_link(tb_node, therapy_node, 'Pallative\nCare\n(Oncology)')

    make_link(therapy_node, ct_node, 'Follow-up', style='solid')
    make_link(therapy_node, pet_node, 'Follow-up', style='solid')

dot_graph.set_overlap(False)
dot_graph.set_rankdir('UD')
dot_graph.write_svg('lungstage_data.svg', prog = 'dot')
SVG('lungstage_data.svg')


# # Other Diagrams

# In[ ]:


dot_graph = pydot.Dot(graph_type='digraph')

def make_node(name,shape):
    cur_node = pydot.Node(name)
    cur_node.set_shape(shape)
    dot_graph.add_node(cur_node)
    return cur_node

def make_link(a_node, b_node, label = None, width = 1, style='dashed'):
    cur_edge = pydot.Edge(a_node,b_node)
    cur_edge.set_penwidth(width)
    cur_edge.set_style(style)
    if label is not None: cur_edge.set_label(label)
    dot_graph.add_edge(cur_edge)
    return cur_edge

mri_node = make_node('mri', 'folder')
mri_node.set_label('MRI Image Data\n(T2, DWI, Scout)')

sort_node = make_node('sort', 'record')
sort_node.set_label('{{Neural Image Sorter}|{DWI|T1|T2|Other}}')

make_link(mri_node, sort_node, style = 'solid')

pos_node = make_node('position', 'record')
pos_node.set_label('{{Neural Position\nEstimator}|{z}}')

make_link(mri_node, pos_node, style = 'solid')

stage_node = make_node('t2_stage', 'record')
stage_node.set_label('{{ Staging}|{T0|T1|T2|T3|T4}|{N0|N1|N2}|{M0|M1}}')

make_link(mri_node, stage_node, 'MRI Images', style = 'solid')

make_link(sort_node, stage_node, 'MRI Category')
make_link(pos_node, stage_node, 'Position\nEstimate')

outcome_node = make_node('outcome', 'record')
outcome_node.set_label('{{Outcome}|{Recurrence|Remission}}')

make_link(mri_node, outcome_node, 'MRI Images', style = 'solid')

make_link(stage_node, outcome_node, 'Stage\nEstimation')
make_link(sort_node, outcome_node)
for inode in [pos_node, sort_node]:
    inode.set_style('filled')
    inode.set_fillcolor('lightblue')

for inode in [stage_node, outcome_node]:
    inode.set_style('filled')
    inode.set_fillcolor('lightgreen')
    

dot_graph.set_rankdir('UD')
dot_graph.write_svg('reading_t2.svg')
dot_graph.write_png('reading_t2.png')
SVG('reading_t2.svg')


# # MRI Project

# In[ ]:


from glob import glob
from skimage.io import imread
from matplotlib import cm
import numpy as np
import os
import pandas as pd
base_dir = '/Users/mader/Dropbox/4Quant/Projects/TumorSegmentation/paper_figures/sample_pat_slices/'
raw_slices = glob(os.path.join(base_dir,'0*_*.tif'))
seg_slices = glob(os.path.join(base_dir,'marked_*.png'))
dot_graph = pydot.Dot(graph_type='digraph')

img_node = lambda im_path, label: """<<TABLE border="0" cellborder="0"><TR><TD width="60" height="50" fixedsize="true"><IMG SRC="{src}" scale="true"/></TD></TR><tr><td><font point-size="12">{label}</font></td></tr></TABLE>>""".format(src=os.path.abspath(im_path), label=label)

t_slice = imread(raw_slices[0])

from scipy.ndimage.morphology import binary_fill_holes
i_slice = imread(seg_slices[0])
bw_slice = binary_fill_holes(i_slice[:,:,0]>i_slice[:,:,1])

from skimage.measure import label
label_slice = label(bw_slice)

from skimage.measure import regionprops
slice_regions = regionprops(label_slice)


flow_figs_dir = 'flowchart_figs'
try:
    os.mkdir(flow_figs_dir)
except:
    print(flow_figs_dir,'already exists')
    
def make_img(in_arr,out_name, **kwargs):
    fig, ax1 = plt.subplots(1,1, figsize = (5,5))
    ax1.imshow(in_arr, interpolation = 'none', **kwargs)
    ax1.axis('off')
    out_path = os.path.join(flow_figs_dir,out_name)
    fig.savefig(out_path)
    plt.close('all')
    return out_path

    
pacs = make_node('PACS', 'box')
pacs.set_label(img_node(make_img(t_slice,'start.png', cmap = 'bone'),'Patient Image'))


dnn = make_node('DNN', 'box3d')
dnn.set_label('Deep Segmentation \n Neural Network')

make_link(pacs, dnn, 'Whitened\nImages', style = 'solid')

make_link(dnn, dnn, 'Continuous\nExpert\nFeedback', style = 'dashed')

seg_img = make_node('Seg_Img', 'box')
seg_img.set_label(img_node(seg_slices[0],'Segmented Image'))

tumor_seg_label = img_node(make_img(bw_slice,'seg.png', cmap = 'bone'),'Segmented Tumor')

make_link(dnn,seg_img, tumor_seg_label , style = 'solid')

make_link(pacs,seg_img, style='solid')

ft_img = make_node('Feature_img', 'folder')
ft_img.set_label(img_node(make_img(label_slice,'labels.png', cmap = cm.gist_earth),'Labeled Features'))

make_link(seg_img, ft_img,'Morphological\nAnalysis',  style='solid')

summ_table = make_node('FinalStat', 'invhouse')

for clabel,c_reg in enumerate(slice_regions):
    cur_feat = make_node('Feature_img_%d' % clabel, 'note')
    cbox = c_reg.bbox
    sub_reg_img = label_slice[cbox[0]:cbox[2],cbox[1]:cbox[3]]
    cur_feat.set_label(img_node(make_img(sub_reg_img,'label_%d.png' % clabel, cmap = cm.gist_earth, vmin = 0, vmax = label_slice.max()),'Feature {}'.format(clabel)))
    make_link(ft_img, cur_feat,  style='dashed')
    
    cur_feat_lab = make_node('Feature_table_%d' % clabel, 'record')
    combo_dict = dict(zip(['Area', 'Perimeter', 'Circularity', 'Diameter', 'Solidity'],
                 [c_reg.area, c_reg.perimeter, c_reg.eccentricity, c_reg.equivalent_diameter, c_reg.solidity]))
    stat_label = pd.DataFrame([combo_dict]).to_html()
    stat_label = '|'.join(['{ %s | %2.2f}' % (rname,rval) for rname, rval in combo_dict.iteritems()])
    cur_feat_lab.set_label('<'+stat_label+'>')
    make_link(cur_feat, cur_feat_lab, style='solid')
    make_link(cur_feat_lab, summ_table, style='solid')

summ_table.set_label('Summary Statistics\nDecision Tree')


tumor_stage_node = make_node('tumor_stage', 'record')
meta_stage_node = make_node('meta_stage', 'record')
node_stage_node = make_node('node_stage', 'record')

t_val = np.random.uniform(0,1, size=5)
t_val *= 100.0/t_val.sum()
m_val = np.random.uniform(0,1, size=2)
m_val *= 100.0/m_val.sum()
n_val = np.random.uniform(0,1, size=4)
n_val *= 100.0/n_val.sum()
tumor_stage_node.set_label('|'.join( ['{T%d | %2.1f%%}' % (i, j) for i,j in enumerate(sorted(t_val))]))

node_stage_node.set_label('|'.join( ['{N%d | %2.1f%%}' % (i, j) for i,j in enumerate(n_val)]))
meta_stage_node.set_label('|'.join( ['{M%d | %2.1f%%}' % (i, j) for i,j in enumerate(m_val)]))

for i in range(len(t_val)): make_link(summ_table, tumor_stage_node, style='solid')
for i in range(len(n_val)): make_link(summ_table, node_stage_node, style='solid')
for i in range(len(m_val)): make_link(summ_table, meta_stage_node, style='solid')

    
dot_graph.set_rankdir('UD')
#dot_graph.set_overlap(False)
dot_graph.write_svg('cnn_morp_tree_proc.svg', prog = 'dot')
SVG('cnn_morp_tree_proc.svg')


# In[ ]:


from skimage.measure import regionprops
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(label_slice,cmap = cm.gist_earth)
all_reg = regionprops(label_slice)
c_reg = all_reg[0]
import StringIO as sio
pd.DataFrame([dict(zip(['Area', 'Perimeter', 'Circularity', 'Diameter', 'Solidity'],
                 [c_reg.area, c_reg.perimeter, c_reg.eccentricity, c_reg.equivalent_diameter, c_reg.solidity]))]).to_html()


# In[ ]:


[c_reg.area, c_reg.perimeter, c_reg.eccentricity, c_reg.equivalent_diameter, c_reg.solidity]


# # Tensorflow Inception Example

# In[ ]:



import os.path
import re
import sys
import tarfile
import cv2
import skimage.transform
import scipy
import matplotlib.pyplot as plt

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob
# pylint: enable=unused-import,g-bad-import-order

from tensorflow.python.platform import gfile

TF_LOGGING = False
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def maybe_download_and_extract(dest_directory):
  """Download and extract model tar file."""
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

my_model_dir = '/Users/mader/Desktop/trained-imagenet'
maybe_download_and_extract(my_model_dir)
with gfile.FastGFile(os.path.join(
  my_model_dir, 'classify_image_graph_def.pb'), 'r') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    resized_arr = np.zeros((299,299,3))
    sub_img = np.expand_dims(resized_arr.astype(np.float32),0)/255.0 # so the values are small
    print(sub_img.shape)
    tf_new_image = tf.Variable(sub_img)
    tf_shift_image = tf.clip_by_value(255*tf_new_image,0,255) - 128 # shift the value to be mean centered and clip the input
    _ = tf.import_graph_def(graph_def, name='', input_map={"Sub:0": tf_shift_image})
    c_graph = graph_def


# In[ ]:


try:
    sess.close()
except:
    print("No session to close")
sess = tf.InteractiveSession()
tf.initialize_all_variables().run(session = sess)
softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')


# In[ ]:


color_words = [] #["weights","biases"]
def add_inputs(graph,last_tens, max_depth = 3):
    if max_depth == 0: return graph
    dst_node = last_tens.name
    for c_input in last_tens.op.inputs:
        src_node = c_input.name
        edge_args = {}
        print(dst_node)
        for cw in color_words: 
            if (src_node.find(cw)>=0): 
                edge_args['color']="#ff0000"
        #if (dst_node=="softmax:0"): 
        #    edge_args['color']="#00ff00" 
        edge = pydot.Edge(src_node, dst_node, **edge_args)
        # and we obviosuly need to add the edge to our graph
        graph.add_edge(edge)
        graph = add_inputs(graph,c_input, max_depth-1)
    return graph
        


# In[ ]:


dot_graph = pydot.Dot(graph_type='digraph')
dot_graph = add_inputs(dot_graph,softmax_tensor, max_depth = 5)
dot_graph.write_svg('inception_net.svg')
SVG('inception_net.svg')


# In[ ]:


pe = pydot.Edge("Input\nImage","mixed_10/join:0",color="#ff0000", style="dashed")
dot_graph.add_edge(pe)
pe = pydot.Edge("softmax:0","softmax_cross_entropy_with_logits", color="#00ff00")
dot_graph.add_edge(pe)
pe = pydot.Edge("jellyfish_vector","softmax_cross_entropy_with_logits")
dot_graph.add_edge(pe)
                  
dot_graph.write_svg('inception_net.svg')
SVG('inception_net.svg')


# In[ ]:


tandy = "bob"
tandy.find('')


# In[ ]:


import cPickle
with open('/Users/mader/Dropbox/4Quant/Projects/TumorSegmentation/tensorflow/cout_graph.pkl','r') as r:
    gph = cPickle.load(r)


# In[ ]:





# In[ ]:





# In[ ]:


gph = pydot.graph_from_dot_data("""
digraph G {
rankdir=TB;
concentrate=True;
node [shape=record];
layer0 [color=red, label=" (Activation)"];
layer1 [color=blue, label=" (Dense)"];
layer1 -> layer0;
layer2 [label=" (Dropout)"];
layer2 -> layer1;
layer3 [color=red, label=" (Activation)"];
layer3 -> layer2;
layer4 [color=blue, label=" (Dense)"];
layer4 -> layer3;
layer5 [label=" (Flatten)"];
layer5 -> layer4;
layer6 [label=" (Dropout)"];
layer6 -> layer5;
layer7 [color=green, label=" (MaxPooling2D)"];
layer7 -> layer6;
layer8 [color=red, label=" (Activation)"];
layer8 -> layer7;
layer9 [color=green, label=" (Convolution2D)"];
layer9 -> layer8;
layer10 [color=red, label=" (Activation)"];
layer10 -> layer9;
layer11 [color=green, label=" (Convolution2D)"];
layer11 -> layer10;
layer12 [label=" (Dropout)"];
layer12 -> layer11;
layer13 [color=green, label=" (MaxPooling2D)"];
layer13 -> layer12;
layer14 [color=red, label=" (Activation)"];
layer14 -> layer13;
layer15 [color=green, label=" (Convolution2D)"];
layer15 -> layer14;
layer16 [color=red, label=" (Activation)"];
layer16 -> layer15;
layer17 [color=green, label=" (Convolution2D)"];
layer17 -> layer16;
layer18 [label=" (Dropout)"];
layer18 -> layer17;
layer19 [color=green, label=" (MaxPooling2D)"];
layer19 -> layer18;
layer20 [color=red, label=" (Activation)"];
layer20 -> layer19;
layer21 [color=green, label=" (Convolution2D)"];
layer21 -> layer20;
layer22 [color=red, label=" (Activation)"];
layer22 -> layer21;
layer23 [color=green, label=" (Convolution2D)"];
layer23 -> layer22;
layer24 [color=red, label=" (Activation)"];
layer24 -> layer23;
}
""")


# In[ ]:


gph.write_svg('test_net.svg',prog = 'neato')
SVG('test_net.svg')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'gph.write_svg')


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1
from keras.layers import LSTM, ActivityRegularization
from keras.layers import merge, Input
from keras.models import Sequential, Graph, Model

def simple_ltsm_model(maxlen,vocab_size,out_size):
    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, return_sequences=False, input_shape=(maxlen, vocab_size)))
    #model.add(Dropout(0.25))
    #model.add(LSTM(1024, return_sequences=False))
    #model.add(Dropout(0.25))
    #model.add(Dense(out_size,  W_regularizer = l1(0.01)))
    model.add(Activation('sigmoid'))
    return model

def medium_ltsm_model(maxlen,vocab_size,out_size, net_depth = 512):
    seq_input = Input((maxlen, vocab_size),name='Sequence of Words')
    ltsm_lay = LSTM(net_depth, return_sequences=False, name='Read Word by Word')(seq_input)
    
    full_input = Input((vocab_size,),name = 'Full Sentence Vector')
    prep_full = Dense(net_depth, activation = 'relu', name='Preprocess Sentence Vector')(full_input)
    
    merge_lay = merge([prep_full,ltsm_lay], mode='concat', concat_axis=1, name='Combine')
    do_lay = Dropout(0.25, name='Randomly Remove Elements')(merge_lay)
    mix_lay = Dense(out_size,  activation = 'sigmoid', name='Mashup All Components')(do_lay)
    model = Model(input=[seq_input,full_input], output=mix_lay)
    return model


# In[ ]:


model = medium_ltsm_model(50,100,100,100)


# In[ ]:


from keras.utils.visualize_util import model_to_dot
from IPython.display import SVG
# Define model
vmod = model_to_dot(model)
vmod.write_svg('se_ltsm.svg')
SVG('se_ltsm.svg')


# In[ ]:





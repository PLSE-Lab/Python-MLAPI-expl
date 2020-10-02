#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt

#plt.ioff()

import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,                            homogeneity_score,completeness_score,silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope


markers = {',': 'pixel', 'o': 'circle','*': 'star', 'v': 'triangle_down',
           '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right', 
           '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right', 
           '8': 'octagon', 's': 'square', 'p': 'pentagon', 
           'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x', '.': 'point', 
           'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline',
           'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 
           1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 5: 'caretright',
           6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 9: 'caretrightbase', 10: 'caretupbase',
           11: 'caretdownbase', 'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}
markers_keys = list(markers.keys())[:20]

font = {'family' : 'normal',
         'weight' : 'bold',
         'size'   : 30}

mpl.rc('font', **font)

sns.set_style("ticks")

colors = ["windows blue", "amber", 
          "greyish", "faded green", 
          "dusty purple","royal blue","lilac",
          "salmon","bright turquoise",
          "dark maroon","light tan",
          "orange","orchid",
          "sandy","topaz",
          "fuchsia","yellow",
          "crimson","cream"
          ]
current_palette = sns.xkcd_palette(colors)

def print_2D( points,label,id_map ):
    '''
    points: N_samples * 2
    label: (int) N_samples
    id_map: map label id to its name
    '''  
    fig = plt.figure()
    #current_palette = sns.color_palette("RdBu_r", max(label)+1)
    n_cell,_ = points.shape
    if n_cell > 500:
        s = 10
    else:
        s = 20
    
    ax = plt.subplot(111)
    print( np.unique(label) )
    for i in np.unique(label):
        ax.scatter( points[label==i,0], points[label==i,1], c=current_palette[i], label=id_map[i], s=s,marker=markers_keys[i] )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
        
    ax.legend(scatterpoints=1,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),ncol=6,
              fancybox=True,
              prop={'size':8}
              )
    sns.despine()
    return fig

def print_heatmap( points,label,id_map ):
    '''
    points: N_samples * N_features
    label: (int) N_samples
    id_map: map label id to its name
    '''
    # = sns.color_palette("RdBu_r", max(label)+1)
    #cNorm = colors.Normalize(vmin=0,vmax=max(label)) #normalise the colormap
    #scalarMap = cm.ScalarMappable(norm=cNorm,cmap='Paired') #map numbers to colors
    
    index = [id_map[i] for i in label]
    df = DataFrame( 
            points,
            columns = list(range(points.shape[1])),
            index = index
            )
    row_color = [current_palette[i] for i in label]
    
    cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
    g = sns.clustermap( df,cmap=cmap,row_colors=row_color,col_cluster=False,xticklabels=False,yticklabels=False) #,standard_scale=1 )
    
    return g.fig

def measure( predicted,true ):
    NMI = normalized_mutual_info_score( true,predicted )
    print("NMI:"+str(NMI))
    RAND = adjusted_rand_score( true,predicted )
    print("RAND:"+str(RAND))
    HOMO = homogeneity_score( true,predicted )
    print("HOMOGENEITY:"+str(HOMO))
    COMPLETENESS = completeness_score( true,predicted )
    print("COMPLETENESS:"+str(COMPLETENESS))
    return {'NMI':NMI,'RAND':RAND,'HOMOGENEITY':HOMO,'COMPLETENESS':COMPLETENESS}

def clustering( points, k=2,name='kmeans'):
    '''
    points: N_samples * N_features
    k: number of clusters
    '''
    if name == 'kmeans':
        kmeans = KMeans( n_clusters=k,n_init=100 ).fit(points)
        ## print within_variance
        #cluster_distance = kmeans.transform( points )
        #within_variance = sum( np.min(cluster_distance,axis=1) ) / float( points.shape[0] )
        #print("AvgWithinSS:"+str(within_variance))
        if len( np.unique(kmeans.labels_) ) > 1: 
            si = silhouette_score( points,kmeans.labels_ )
            #print("Silhouette:"+str(si))
        else:
            si = 0
            print("Silhouette:"+str(si))
        return kmeans.labels_,si
    
    if name == 'spec':
        spec= SpectralClustering( n_clusters=k,affinity='cosine' ).fit( points )
        si = silhouette_score( points,spec.labels_ )
        print("Silhouette:"+str(si))
        return spec.labels_,si
        
def cart2polar( points ):
    '''
    points: N_samples * 2
    '''
    return np.c_[np.abs(points), np.angle(points)]
        
def outliers_detection(expr):
    x = PCA(n_components=2).fit_transform(expr)
    ee = EllipticEnvelope()
    ee.fit(x)
    oo = ee.predict(x)
    
    return oo


# In[ ]:


# -*- coding: utf-8 -*-
from keras.layers import Input,Dense,Activation,Lambda,RepeatVector,concatenate, multiply, dot,                             Reshape,Layer,Dropout,BatchNormalization,Permute
import keras.backend as K
from keras.models import Model
#from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.utils.layer_utils import print_summary
import numpy as np
from keras.optimizers import RMSprop,Adagrad,Adam
from keras import metrics
import h5py

tau = 1.0

def sampling(args):
    epsilon_std = 1.0
    
    if len(args) == 2:
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), 
                              mean=0.,
                              stddev=epsilon_std)
    #
        return z_mean + K.exp( z_log_var / 2 ) * epsilon
    else:
        z_mean = args[0]
        epsilon = K.random_normal(shape=K.shape(z_mean), 
                              mean=0.,
                              stddev=epsilon_std)
        return z_mean + K.exp( 1.0 / 2 ) * epsilon
        
        
def sampling_gumbel(shape,eps=1e-8):
    u = K.random_uniform( shape )
    return -K.log( -K.log(u+eps)+eps )

def compute_softmax(logits,temp):
    z = logits + sampling_gumbel( K.shape(logits) )
    return K.softmax( z / temp )

def gumbel_softmax(args):
    logits,temp = args
    return compute_softmax(logits,temp)

class NoiseLayer(Layer):
    def __init__(self, ratio, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.ratio = ratio

    def call(self, inputs, training=None):
        def noised():
            return inputs * K.random_binomial(shape=K.shape(inputs),
                                              p=self.ratio
                                              )
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'ratio': self.ratio}
        base_config = super(NoiseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


        return dict(list(base_config.items()) + list(config.items()))
    


# In[ ]:


class VASC:
    def __init__(self,in_dim,latent=2,var=False):
        self.in_dim =in_dim
        self.vae = None
        self.ae = None
        self.aux = None
        self.latent = latent
        self.var = var
        
    
    def vaeBuild( self ):
        var_ = self.var
        in_dim = self.in_dim
        expr_in = Input( shape=(self.in_dim,) )
        
        ##### The first part of model to recover the expr. 
        h0 = Dropout(0.5)(expr_in) 
        ## Encoder layers
        h1 = Dense( units=512,name='encoder_1',kernel_regularizer=regularizers.l1(0.01) )(h0)
        h2 = Dense( units=128,name='encoder_2' )(h1)
        h2_relu = Activation('relu')(h2)
        h3 = Dense( units=32,name='encoder_3' )(h2_relu)
        h3_relu = Activation('relu')(h3)

        
        z_mean = Dense( units= self.latent ,name='z_mean' )(h3_relu)
        if self.var:
            z_log_var = Dense( units=2,name='z_log_var' )(h3_relu)
            z_log_var = Activation( 'softplus' )(z_log_var)
       
                    
        ## sampling new samples
            z = Lambda(sampling, output_shape=(self.latent,))([z_mean,z_log_var])
        else:
            z = Lambda(sampling, output_shape=(self.latent,))([z_mean])
        
        ## Decoder layers
        decoder_h1 = Dense( units=32,name='decoder_1' )(z)
        decoder_h1_relu = Activation('relu')(decoder_h1)
        decoder_h2 = Dense( units=128,name='decoder_2' )(decoder_h1_relu)
        decoder_h2_relu = Activation('relu')(decoder_h2)  
        decoder_h3 = Dense( units=512,name='decoder_3' )(decoder_h2_relu)
        decoder_h3_relu = Activation('relu')(decoder_h3)
        expr_x = Dense(units=self.in_dim,activation='sigmoid')(decoder_h3_relu)

        
        expr_x_drop = Lambda( lambda x: -x ** 2 )(expr_x)
        #expr_x_drop_log = concatenate( [drop_ratio,expr_x_drop] )  ###  log p_drop =  log(exp(-\lambda x^2))
        expr_x_drop_p = Lambda( lambda x:K.exp(x) )(expr_x_drop)
        expr_x_nondrop_p = Lambda( lambda x:1-x )( expr_x_drop_p )
        expr_x_nondrop_log = Lambda( lambda x:K.log(x+1e-20) )(expr_x_nondrop_p)
        expr_x_drop_log = Lambda( lambda x:K.log(x+1e-20) )(expr_x_drop_p)        
        expr_x_drop_log = Reshape( target_shape=(self.in_dim,1) )(expr_x_drop_log)
        expr_x_nondrop_log = Reshape( target_shape=(self.in_dim,1) )(expr_x_nondrop_log)
        logits = concatenate( [expr_x_drop_log,expr_x_nondrop_log],axis=-1 )
#         print(logits.shape)
        temp_in = Input( shape=(self.in_dim,) )
        temp_ = RepeatVector( 2 )(temp_in)
#         print(temp_.shape)
        temp_ = Permute( (2,1) )(temp_)
        samples = Lambda( gumbel_softmax,output_shape=(self.in_dim,2,) )( [logits,temp_] )          
        samples = Lambda( lambda x:x[:,:,1] )(samples)
        samples = Reshape( target_shape=(self.in_dim,) )(samples)      
#         print(samples.shape, expr_x)
        
#         out = dot( [expr_x,samples], axes=-1 )
        out = multiply( [expr_x,samples] )
        class VariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(VariationalLayer, self).__init__(**kwargs)
        
            def vae_loss(self, x, x_decoded_mean):
                xent_loss = in_dim * metrics.binary_crossentropy(x, x_decoded_mean)
                if var_:
                    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                else:
                    kl_loss = - 0.5 * K.sum(1 + 1 - K.square(z_mean) - K.exp(1.0), axis=-1)
                return K.mean(xent_loss + kl_loss)
        
            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean = inputs[1]
                loss = self.vae_loss(x, x_decoded_mean)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x
        
        y = VariationalLayer()([expr_in, out])
        vae = Model( inputs= [expr_in,temp_in],outputs=y )
        
        opt = RMSprop( lr=0.001 )
        vae.compile( optimizer=opt,loss=None )
        
        ae = Model( inputs=[expr_in,temp_in],outputs=[ h1,h2,h3,h2_relu,h3_relu,
                                                       z_mean,z,decoder_h1,decoder_h1_relu,
                                                       decoder_h2,decoder_h2_relu,decoder_h3,decoder_h3_relu,
                                                       samples,out
                                                       ] )
        aux = Model( inputs=[expr_in,temp_in],outputs=[out] )
        
        self.vae = vae
        self.ae = ae
        self.aux = aux


# In[ ]:


def vasc( expr,
          epoch = 5000,
          latent=2,
          patience=50,
          min_stop=500,
          batch_size=32,
          var = False,
          prefix='test',
          label=None,
          log=True,
          scale=True,
          annealing=False,
          tau0 = 1.0,
          min_tau = 0.5,
          rep=0):
    '''
    VASC: variational autoencoder for scRNA-seq datasets
    
    ============
    Parameters:
        expr: expression matrix (n_cells * n_features)
        epoch: maximum number of epochs, default 5000
        latent: dimension of latent variables, default 2
        patience: stop if loss showes insignificant decrease within *patience* epochs, default 50
        min_stop: minimum number of epochs, default 500
        batch_size: batch size for stochastic optimization, default 32
        var: whether to estimate the variance parameters, default False
        prefix: prefix to store the results, default 'test'
        label: numpy array of true labels, default None
        log: if log-transformation should be performed, default True
        scale: if scaling (making values within [0,1]) should be performed, default True
        annealing: if annealing should be performed for Gumbel approximation, default False
        tau0: initial temperature for annealing or temperature without annealing, default 1.0
        min_tau: minimal tau during annealing, default 0.5
        rep: not used
    
    =============
    Values:
        point: dimension-*latent* results
        A file named (*prefix*_*latent*_res.h5): we prefer to use this file to analyse results to the only return values.
        This file included the following keys:
            POINTS: all intermediated latent results during the iterations
            LOSS: loss values during the training procedure
            RES*i*: i from 0 to 14
                - hidden values just for reference
        We recommend use POINTS and LOSS to select the final results in terms of users' preference.
    '''
    
    
    expr[expr<0] = 0.0

    if log:
        expr = np.log2( expr + 1 )
    if scale:
        for i in range(expr.shape[0]):
            expr[i,:] = expr[i,:] / np.max(expr[i,:])
            
#     if outliers:
#        o = outliers_detection(expr)
#        expr = expr[o==1,:]
#        if label is not None:
#            label = label[o==1]
    
    
    if rep > 0:
        expr_train = np.matlib.repmat( expr,rep,1 )
    else:
        expr_train = np.copy( expr )
    
    vae_ = VASC( in_dim=expr.shape[1],latent=latent,var=var )
    vae_.vaeBuild()
    #print_summary( vae_.vae )
    
    points = []
    loss = []
    prev_loss = np.inf
    #tau0 = 1.
    tau = tau0
    #min_tau = 0.5
    anneal_rate = 0.0003
    for e in range(epoch):
        cur_loss = prev_loss
        
        #mask = np.ones( expr_train.shape,dtype='float32' )
        #mask[ expr_train==0 ] = 0.0
        if e % 100 == 0 and annealing:
            tau = max( tau0*np.exp( -anneal_rate * e),min_tau   )
            print(tau)

        tau_in = np.ones( expr_train.shape,dtype='float32' ) * tau
        #print(tau_in.shape)
        
        loss_ = vae_.vae.fit( [expr_train,tau_in],expr_train,epochs=1,batch_size=batch_size,
                             shuffle=True,verbose=0
                             )
        train_loss = loss_.history['loss'][0]
        cur_loss = min(train_loss,cur_loss)
        loss.append( train_loss )
        #val_loss = -loss.history['val_loss'][0]
        res = vae_.ae.predict([expr,tau_in])
        points.append( res[5] )
        if label is not None:
            k=len(np.unique(label))
            
        if e % patience == 1:
            print( "Epoch %d/%d"%(e+1,epoch) )
            print( "Loss:"+str(train_loss) )
            if abs(cur_loss-prev_loss) < 1 and e > min_stop:
                break
            prev_loss = train_loss
            if label is not None:
                try:
                    cl,_ = clustering( res[5],k=k )
                    measure( cl,label )
                except:
                    print('Clustering error')    
                    
    #
    ### analysis results
    #cluster_res = np.asarray( cluster_res )
    points = np.asarray( points )
    aux_res = h5py.File( prefix+'_'+str(latent)+'_res.h5',mode='w' )
    #aux_res.create_dataset( name='EXPR',data=expr )
    #aux_res.create_dataset( name='CLUSTER',data=cluster_res )
    aux_res.create_dataset( name='POINTS',data=points )
    aux_res.create_dataset( name='LOSS',data=loss )
    count = 0
    for r in res:
        aux_res.create_dataset( name='RES'+str(count),data=r)
        count += 1
    aux_res.close()
    
    return res[5]


# In[ ]:


config={
    'epoch':100,
    'batch_size':32,
    'latent':2,
    'log':False,
    'scale':True,
    'patience':50
}


# In[ ]:


filename = '../input/biase.txt'
data = open( filename )
head = data.readline().rstrip().split()

label_file = open( '../input/biase_label.txt' )
label_dict = {}
for line in label_file:
    temp = line.rstrip().split()
    label_dict[temp[0]] = temp[1]
label_file.close()

label = []
for c in head:
    if c in label_dict.keys():
        label.append(label_dict[c])
    else:
        print(c)

label_set = []
for c in label:
    if c not in label_set:
        label_set.append(c)
name_map = {value:idx for idx,value in enumerate(label_set)}
id_map = {idx:value for idx,value in enumerate(label_set)}
label = np.asarray( [ name_map[name] for name in label ] )

expr = []
for line in data:
    temp = line.rstrip().split()[1:]
    temp = [ float(x) for x in temp]
    expr.append( temp )

expr = np.asarray(expr).T
n_cell,_ = expr.shape


# In[ ]:


expr.shape, label.shape, label


# In[ ]:


for i in range(10):
    print("Iteration:"+str(i))
    res = vasc( expr,var=False,
                latent=config['latent'],
                annealing=False,
                batch_size=config['batch_size'],
                label=label,
                scale=config['scale'],
                patience=config['patience'],
            )
#            res_file = PREFIX+'_res.h5'
#            res_data = h5py.File( name=res_file,mode='r' )
#            dim2 = res_data['RES5']
#            print(np.max(dim2))

    print(res.shape)
    k = len( np.unique(label) )
    cl,_ = clustering( res,k=k)
    dm = measure( cl,label )

#            res_data.close()
    ### analysis results
    # plot loss

    # plot 2-D visulation
    fig = print_2D( points=res,label=label,id_map=id_map )
#        fig.savefig('embryo.eps')
#        fig = print_2D( points=res_data['RES5'],label=label,id_map=id_map )
#        fig.show()
#        res_data.close()
#        time.sleep(30)
    #res_data.close()
# plot NMI,ARI curve
#    
#    pollen = h5py.File( name=DATASET+'_'+str(latent)+'_.h5',mode='w' )
#    pollen.create_dataset( name='NMI',data=nmi)
#    pollen.create_dataset( name='ARI',data=ari )
#    pollen.create_dataset( name='HOM',data=hom )
#    pollen.create_dataset( name='COM',data=com )
#    pollen.close()
#    

#print("============SUMMARY==============")
#k = len(np.unique(label))
#for r in res:
#    print("======"+str(r.shape[1])+"========")
#    pred,si = clustering( r,k=k )
#    if label is not None:
#        metrics = measure( pred,label )


# In[ ]:


# expr = np.exp(expr) - 1 
# expr = expr / np.max(expr)

# percentage = [0.5]

# for j in range(1):
#     print(j)
#     p = percentage[j]
#     samples = np.random.choice( n_cell,size=int(n_cell*p),replace=True )
#     expr_train = expr[ samples,: ]
#     label_train = label[samples]


# In[ ]:





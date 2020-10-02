#!/usr/bin/env python
# coding: utf-8

# # Recommending Music Artists with Boltzmann Machines
# ### Release 1.0 - June, 2020
# ### Paulo Breviglieri

# ## Foreword

# <p style="text-align: justify">This notebook describes how a simple <b>Boltzmann machine</b> can assist with <b>music artist recommendations</b> based on artist popularity records collected by <b>last.fm</b>, a UK-based music website and app that supplies recommendations based on detailed user profiles constructed from 'scrobbles' - tracks listened to from diverse music streaming apps, like Spotify and SoundHound, and internet radio stations.</p>
# <p style="text-align: justify">The case study described herein has an educational nuance and aims at highlighting how a simple <b>unsupervised learning</b> solution addresses a specific problem. Actual commercial recommendation platforms are incredibly more powerful, sophisticated, comprehensive and rely on more advanced artificial intelligence technologies. Positioning the Boltzmann machine developed here as an alternative for a real world implementation is definitely not an intent of this exercise.</p>
# <p style="text-align: justify">The author also highlights specific singularities of this notebook not usually found in traditional machine learning exercises:</p>
# <ul>
#     <li style="text-align: justify">Instead of being built upon existing machine learning frameworks, the Boltzmann machine is <b>hard coded as a Python class</b>;</li>
#     <li style="text-align: justify">In addition to the traditional performance assessment based on accuracy and loss metrics, a <b>subjective test</b> is performed at the end to verify the appropriateness of actual music artist recommendations delivered by the Boltzmann machine to two specific users with different music gender preferences.</li>
# </ul>
# <p style="text-align: justify">Section 1 includes a summarized, brief mathematical description of Boltzmann machines, not intented to be exhaustive and comprehensive at all. Readers interested only in the practical implementation may skip it.</p>
# <p style="text-align: justify">The latest code release may be found in the author's <a href="https://github.com/pcbreviglieri">GitHub</a> repository. Logic enhancement and code forking are welcome and encouraged by the author, provided that this work is properly referenced. Thank you.</p>

# ## 1. Introduction

# ### 1.1. Unsupervised learning

# <p style="text-align: justify">Unlike supervised learning, where machines are typically fed with pairs of known inputs and corresponding outputs to direct the learning and self-adaptation processes, in unsupervised learning machines are supplied with unlabeled responses and are expected to learn probability distributions from inference, with minimum or no human guidance. As a result, unsupervised learning machines have distinct architectures and functional principles. Traditional concepts found in supervised learning, like feedforwarding and input / output layers, are not applicable - or have different interpretations - in unsupervised learning.</p>

# ### 1.2. Boltzmann machines

# <p style="text-align: justify">Botzmann machines are unsupervised learning neural networks pertaining to a family of solutions not as wide and popularly explored as others: <b>energy based models</b>. Put simply, in energy based models the system under analysis will always self-adapt to changes in its constituent elements in pursuit of the lowest possible overall compounded energy state. This concept is central in several fields of Physics, from thermodynamycs to quantum mechanics.</p>
# <p style="text-align: justify">A traditional Boltzmann machine comprises a finite set of neurons interconnected with each other.</p>
# <p style="text-align: justify">Neurons- or <b>nodes</b> - are in turn subdivided into two specific sets: <b>visible</b> and <b>hidden</b>, as described in the picture below.</p>

# <img src="https://i.imgur.com/u5G1z1L.png" width="500" height="100">

# <p style="text-align: justify">In an analogy with the human body, the author likes to think of <b>visible nodes</b> as the neural terminations we have in our eyes, nose, tongue and skin to capture external stimulus such as images, sounds, smells, tastes and sensations like heat, cold and pain. Accordingly, <b>hidden nodes</b> might be interpreted as the set of neurons in our brains responsible to process the information received from the neural terminations and take action - always aiming at self-adapting and conducting the organism to the most appropriate state - the 'lowest energy' state - for a given set of external circumstances.</p>
# <p style="text-align: justify">A pictorial example: if the neurons located in the optic nerves on and behind our eyes' retina (the 'visible nodes') experience a sudden increase in external light intensity, this information is quickly delivered to the neurons located in the brain's occipital lobe (the 'hidden nodes') who, in turn, process this information and 'instruct' the eye muscles to constrict pupils, so that the amount of light  reaching the retina is reduced and retina cells are preserved. The overall organism optical network is thus taken to a new optimal 'state' that will last until the environment perceived by the visible nodes change again.</p>

# <p style="text-align: justify">In a Boltzmann machine, all nodes are connected to each other. However, the adoption of such architecture in practical implementations demands higher processing and memory assigment capabilities, reason why a simplified version of Boltzmann machines - so called Restricted Boltzmann Machines (RBMs) - gained traction and became widely used. In RBMs, connections between visible nodes and hidden nodes are maintained, while connections among visible nodes and connections among hidden nodes are eliminated, as illustrated below.</p>

# <img src="https://i.imgur.com/kNI7eLL.png" width="500" height="100">

# <p style="text-align: justify">The relationship between visible and hidden nodes is governed by the weights of the corresponding connections. In addition, biases are assigned to both hidden ('a') and visible ('b') nodes.</p>

# <img src="https://i.imgur.com/G0DvbUp.png" width="500" height="100">

# <p style="text-align: justify">As energy based models in which the objective will always be to self-adapt pursuing the lowest scalar energy state possible, RBMs are governed by one fundamental equation describing the total 'energy' of the network in terms of the values ('states') of visible (v) and hidden (h) nodes along with weights (w) and biases (a, b), all expressed as tensors:</p>

# <p style="text-align: center; font-size: 20px">${E(v,h)} = - \sum \limits _{i}  a_{i}  v_{i} - \sum \limits _{j}  b_{j}  h_{j} - \sum \limits _{i,j}  v_{i}  h_{j} w_{ij} $</p>

# <p style="text-align: justify">Over training, the RBM weights and biases will be adjusted in order to minimize the overall network energy.</p>
# <p style="text-align: justify">Another relevant aspects of RBMs is their nature: RBMs are probabilistic models. Remember that at any moment the RBM will be in a particular state given by the values stored in visible and hidden nodes (neurons), linked by weights and biases. The model will operate based on the probability that a certain state of v and h can be observed. In mathematical terms, such probability will be governed by a joint  distribution - the Boltzmann Distribution, after which this type of learning machines is named:</p>

# <p style="text-align: center; font-size: 20px">${p(v,h)} = \frac{1}{Z} e^{-E(v,h)}$</p>
# <p style="text-align: center">where Z, the 'partition function', is given by:</p>
# <p style="text-align: center; font-size: 20px">${Z} = \sum \limits _{v,h}  e^{-E(v,h)}$</p>

# <p style="text-align: justify"> In Physics, the Boltzmann distribution furnishes the probability of a particle being observed in a given state with energy E. In a RBM, we are interested in the probability to observe a state of v and h based on the overall model energy. As the calculation of the joint Boltzmann probability would be complex in networks with a large numbers of combinations of visible and hidden node values (v and h), the analysis focuses instead on the calculation of the conditional probabilities of hidden nodes being in a particular state <b>given</b> the state of the visible nodes, denoted as <b>p(h|v)</b>, and also the conditional probabilities of visible nodes being in a particular state <b>given</b> the state of the hidden nodes, denoted as <b>p(v|h)</b>:</p>

# <p style="text-align: center; font-size: 20px">${p (h | v)} = \prod  \limits _{i} {p (h _{i} | v)}$</p>
# <p style="text-align: center; font-size: 20px">${p (v | h)} = \prod  \limits _{i} {p (v _{i} | h)}$</p>

# <p style="text-align: justify">In RBMs the values assigned to neurons (node) are <b>binary</b>. In other words, in RBMs we deal with 'activated' and 'non activated' states for both visible and hidden nodes.</p>
# <p style="text-align: justify">This fact allows us to derive the conditional probabilities above for the cases of hidden nodes assuming a value equal to 1 (given visible nodes at certain states) and also the conditional probabilities for the cases of visible nodes assuming a value equal to 1 (given hidden nodes at certain states). After applying the Bayes rule to conditional probabilities, we obtain:</p>

# <p style="text-align: center; font-size: 20px">${p (h_{j}=1 | v)} = \frac {1}{1 + e^{(- (b_{j} + W_{j} v_{i}))}} = \sigma (b_{j} + \sum \limits _{i} v_{i} w_{ij})$</p>
# <p style="text-align: center; font-size: 20px">${p (v_{i}=1 | h)} = \frac {1}{1 + e^{(- (a_{i} + W_{i} h_{j}))}} = \sigma (a_{i} + \sum \limits _{j} h_{j} w_{ij})$</p>

# <p style="text-align: justify">where $\sigma$ is our well known sigmoid function!</p>

# ### 1.3. Training Boltzmann machines

# <p style="text-align: justify">RBMs are trained in a very unique, two-step approach.  Details may be found in "<em><b>A fast learning algorithm for deep belief nets</b></em>" (G.E. Hinton, S. Osindero, Department of Computer Science, University of Toronto, YW. Teh, Department of Computer Science, National University of Singapore), in which Dr. Hinton and his co-authors describe the use of 'complementary priors' to "derive a fast, greedy algorithm that can learn deep, directed belief networks".</p>

# #### 1.3.1. Step 1 - Gibbs sampling

# <p style="text-align: justify">This is an interactive process comprising the following steps, as pictured below:</p>
# <ul>
#     <li style="text-align: justify">An input tensor $v_{0}$ containing binary constituent elements (1's and 0's) of a given observation is fed into visible nodes;</li>
#     <li style="text-align: justify">The activation of hidden nodes, given this input tensor, is predicted via $p(h|v_{0})$</li>
#     <li style="text-align: justify">A new activation of visible nodes, given the previous activation of hidden nodes, is predicted via $p(v|h)$</li>
#     <li style="text-align: justify">The two last steps are repeated k times until a last activation of visible nodes $v_{k}$ is predicted.</li>
# </ul>

# <img src="https://i.imgur.com/iCRhhLs.png" width="500" height="100">

# #### 1.3.2. Step 2 - Contrastive divergence

# <p style="text-align: justify">Updating weights (and biases) is the primary objective of any neural network training program.</p>
# <p style="text-align: justify">In the case of RBMs, the weight tensor is updated through a method called Contrastive Divergence. In a summarized fashion, the activation probabilities of hidden node tensors $h_{0}$ and $h_{k}$ are calculated from visible node tensors $v_{0}$ and $v_{k}$. The difference between the <b>outer products</b> of such activation probabilities with input tensors $v_{0}$ and $v_{k}$ will lead to an updated version of the weight tensor:</p>

# <p style="text-align: center; font-size: 20px">$\Delta W = v_{0} \otimes {p (h_{0} | v_{0})} - v_{k} \otimes {p (h_{k} | v_{k})}$</p>

# <p style="text-align: justify">At last, a new set of updated weights at step 'm' can be estimated with gradient ascent:</p>

# <p style="text-align: center; font-size: 20px">$W_{m} = W_{m-1} + \Delta W$</p>

# ## 2. Objectives of this deep learning exercise

# <p style="text-align: justify">The primary goal of this work is educational, as is the norm in most of the author's notebooks.</p>
# <p style="text-align: justify">A Restricted Boltzmann Machine (RBM) is used to provide music artist recommendations to a particular individual based on:</p>
# <ul>
#     <li style="text-align: justify">Music artist popularity records generated by a multitude of platform users and maintained by last.fm;</li>
#     <li style="text-align: justify">The set of preferred artists enjoyed by one specific individual.</li>
# </ul>
# <p style="text-align: justify">In other words, the machine will identify the subset of music artists appreciated by a particular user (artists A, B, C and D, for example) and offer a second subset of music artists this particular user might be interested in (artists W, X, Y and Z), based on his/her preferences.</p>
# <p style="text-align: justify">Specific expertise will be developed in this exercise, including: </p>
# <ul>
#     <li style="text-align: justify">The construction of a RBM in the form of a Python class that will be later instantiated;</li>
#     <li style="text-align: justify">The generation of predictions for two specific users, with different musical preferences, in addition to the crude (and cold) performance assessment procedures based on the quantification of error metrics;</li>
#     <li style="text-align: justify">The use of PyTorch as the deep learning framework of choice.</li>
# </ul>
# <p style="text-align: justify">A special note on the framework selection. Comparing competitive machine learning frameworks is <em><b>NOT</b></em> an objective herein. Instead, the goal is to highlight that similar frameworks may serve the same purpose regardless of their popularity. PyTorch was adopted in this particular case simply because it offered the author a straightforward coding path that might as well be delivered by other frameworks. No performance assessment guided this selection.</p>

# ### 2.1. Experimenting an alternative approach with Boltzmann machines

# <p style="text-align: justify">As discussed in Section 1.2, Boltzmann machines are probabilistic models. Ideally, visible and hidden nodes would be dynamically activated (1) or not (0), representing an immense multitude of different machine states.</p>
# <p style="text-align: justify">However, the last.fm dataset contains total numbers of scrobbles generated per user, per music artist - a direct measure of artist popularity we intend to explore with the RBM. Furthermore, the dataset includes:</p>
# <ul>
#     <li style="text-align: justify">Both heavy users, who have generated dozens, even hundreds of thousands of 'scrobbles' for selected artists, and light users, who produced few 'scrobbles' for few artists. The model must ideally not allow that heavy user preferences overshadow those of light users;</li>
#     <li style="text-align: justify">For every user, the number of scrobbles per artist may vary from one to thousands. The model shall ideally not allow high scrobble counts per user to drastically overshadow low scrobble counts per user.</li>
# </ul>
# <p style="text-align: justify">Having said that, instead of feeding the model with simple binary inputs, which would correspond to a set of 'scrobbled' and 'not scrobbled' music artists for a given user, the machine will be supplied with scaled inputs ranging from 0 to 1 for each 'scrobbled' music artist, on a per user basis. For a given user, a scaled number of scrobbles of a given artist close to '0' means that a small number of scrobbles was generated - the user briefly checked on that artist but the return rate was none or very small, a sign of low 'popularity' of that artist for that user. Inversely, a scaled number of scrobbles of a given artist, for a given user, close to '1' means that a large number of scrobbles was collected - the user listened to that artist several times with a high return rate, an indication of high 'popularity' of that artist for that user.</p>
# <p style="text-align: justify">The traditional approach (i.e. assigning 1 to 'scrobbled' artists and '0' to 'not scrobbled' artists) would not bring to the analysis how popular a given artist is for a given user. An artist with 1 scrobble would be given the same weight as an artist with, let's say, 1,000 scrobbles. Let's see how the machine performs under these assumptions.</p>
# <p style="text-align: justify">The selected approach is portrayed below.</p>

# <img src="https://i.imgur.com/mZ8wjUz.png" width="500" height="100">

# ## 3. Initial setup

# ### 3.1. Importing required libraries

# <p style="text-align: justify">Along with traditional libraries imported for tensor manipulation and mathematical operations, <a href="https://pytorch.org/">PyTorch</a> is used in this exercise.</p>

# In[ ]:


import numpy as np
import pandas as pd
import torch
from datetime import datetime

start_time = datetime.now()


# ### 3.2. Hard coding a Boltzmann Machine

# <p style="text-align: justify">The creation of machine learning models usually relies on the instantiation of predefined classes provided by frameworks such as PyTorch and Keras. In other words, a 'model' is coded in Python or R as an object - an instance of predefined classes. This allows for the use of the class implicit features and methods, making the coder's task much simpler and the final code itself cleaner and shorter.</p>
# <p style="text-align: justify">As anticipated in Section 2, a different approach is proposed here. A Python class named 'RestrictedBoltzmannMachine' will be developed and further instantiated for the creation of a RBM model.</p>
# <p style="text-align: justify">The 'RestrictedBoltzmannMachine' class comprises the following elements:</p>
# <ol>
#     <li style="text-align: justify">An initialization module where the inherent tensors for weights and biases are defined;</li>
#     <li style="text-align: justify">Two methods (used internally by other methods) devoted to Gibbs sampling as described in Section 1.3.1; </li>
#     <li style="text-align: justify">One method devoted to the model training where, over several epochs and for several batches:</li>
#     <ul>
#         <li style="text-align: justify">Contrastive divergence is executed in 10 rounds ;</li>
#         <li style="text-align: justify">Weight (W) and biases (a and b) tensors are updated; </li>
#         <li style="text-align: justify">Losses are calculated;</li>
#     </ul>
#     <li style="text-align: justify">One method devoted to the model testing where test observations are fed into the RBM and compounded loss metrics calculated;</li>
#     <li style="text-align: justify">One method devoted to predicting recommendations for one particular observation (last.fm user);</li>
# </ol>
# <p style="text-align: justify">Please refer to docstrings for information on the machine structure and functionality.</p>

# In[ ]:


class RestrictedBoltzmannMachine():
    """
    Python implementation of a Restricted Boltzmann Machine (RBM) with 'c_nh' hidden nodes and 'c_nv' visible nodes.
    """
    def __init__(self, c_nv, c_nh):
        """
        RBM initialization module where three tensors are defined:
        W - Weight tensor
        a - Visible node bias tensor
        b - Hidden node bias tensor
        a and b are created as two-dimensional tensors to accommodate batches of observations over training.
        """
        self.W = torch.randn(c_nh, c_nv)
        self.a = torch.randn(1, c_nh)
        self.b = torch.randn(1, c_nv)

        
    def sample_h(self, c_vx):
        """
        Method devoted to Gibbs sampling probabilities of hidden nodes given visible nodes - p (h|v)
        c_vx - Input visible node tensor
        """
        c_w_vx = torch.mm(c_vx, self.W.t())
        c_activation = c_w_vx + self.a.expand_as(c_w_vx)
        c_p_h_given_v = torch.sigmoid(c_activation)
        return c_p_h_given_v, torch.bernoulli(c_p_h_given_v)

    
    def sample_v(self, c_hx):
        """
        Method devoted to Gibbs sampling probabilities of visible nodes given hidden nodes - p (v|h)
        c_hx - Input hidden node tensor
        """
        c_w_hx = torch.mm(c_hx, self.W)
        c_activation = c_w_hx + self.b.expand_as(c_w_hx)
        c_p_v_given_h = torch.sigmoid(c_activation)
        return c_p_v_given_h, torch.bernoulli(c_p_v_given_h)

    
    def train(self, c_nr_observations, c_nr_epoch, c_batch_size, c_train_tensor, c_metric):
        """
        Method through which constrative divergence-based training is performed.
        c_nr_observations - Number of observations used for training
        c_nr_epoch - Number of training epochs
        c_batch_size - Batch size
        c_train_tensor - Tensor containing training observations
        c_metric - Training performance metric of choice ('MAbsE' for Mean Absolute Error, 'RMSE' for Root Mean Square Error)
        """
        print('Training...')
        for c_epoch in range(1, c_nr_epoch + 1):
            c_start_time = datetime.now()
            print(f'Epoch {str(c_epoch)} of {str(c_nr_epoch)} ', end='')
            c_train_loss = 0
            c_s = 0.
            for c_id_user in range(0, c_nr_observations - c_batch_size, c_batch_size):
                c_v0 = c_train_tensor[c_id_user:c_id_user+c_batch_size]
                c_vk = c_train_tensor[c_id_user:c_id_user+c_batch_size]
                c_ph0,_ = self.sample_h(c_v0)
                for c_k in range(10):
                    _,c_hk = self.sample_h(c_vk)
                    _,c_vk = self.sample_v(c_hk)
                    c_vk[c_v0<0] = c_v0[c_v0<0]
                c_phk,_ = self.sample_h(c_vk)
                self.W += (torch.mm(c_v0.t(), c_ph0) - torch.mm(c_vk.t(), c_phk)).t()
                self.b += torch.sum((c_v0 - c_vk), 0)
                self.a += torch.sum((c_ph0 - c_phk), 0)
                if c_metric == 'MAbsE':
                    c_train_loss += torch.mean(torch.abs(c_v0[c_v0>=0] - c_vk[c_v0>=0]))
                elif c_metric == 'RMSE':
                    c_train_loss += np.sqrt(torch.mean((c_v0[c_v0>=0] - c_vk[c_v0>=0])**2))
                c_s += 1.
            c_end_time = datetime.now()
            c_time_elapsed = c_end_time - c_start_time
            c_time_elapsed = c_time_elapsed.total_seconds()
            print(f'- Loss ({c_metric}): {c_train_loss/c_s:.8f} ({c_time_elapsed:.2f} seconds)')


    def test(self, c_nr_observations, c_train_tensor, c_test_tensor, c_metric):
        """
        Method through which testing is performed.
        c_nr_observations - Number of observations used for testing
        c_train_tensor - Tensor containing training observations
        c_test_tensor - Tensor containing testing observations
        c_metric - Training performance metric of choice ('MAbsE' for Mean Absolute Error, 'RMSE' for Root Mean Square Error)
        """
        print('Testing...')
        c_test_loss = 0
        c_s = 0.
        for c_id_user in range(c_nr_observations):
            c_v = c_train_tensor[c_id_user:c_id_user+1]
            c_vt = c_test_tensor[c_id_user:c_id_user+1]
            if len(c_vt[c_vt>=0]) > 0:
                _,c_h = self.sample_h(c_v)
                _,c_v = self.sample_v(c_h)
                if c_metric == 'MAbsE':
                    c_test_loss += torch.mean(torch.abs(c_vt[c_vt>=0] - c_v[c_vt>=0]))
                elif c_metric == 'RMSE':
                    c_test_loss += np.sqrt(torch.mean((c_vt[c_vt>=0] - c_v[c_vt>=0])**2))
                c_s += 1.
        print(f'Test loss ({c_metric}): {c_test_loss/c_s:.8f}')
        
        
    def predict(self, c_visible_nodes):
        """
        Method through which predictions for one specific observation are derived.
        c_visible_nodes - Tensor containing one particular observation (set of values for each visible node) 
        """
        c_h_v,_ = self.sample_h(c_visible_nodes)
        c_v_h,_ = self.sample_v(c_h_v)
        return c_v_h


# ### 3.3. Creating purposed functions

# <p style="text-align: justify">Two specific customized functions address specific needs:</p>
# <ul>
#     <li style="text-align: justify">'<b><em>convert</em></b>' essentially takes the original last.fm dataset table and produces a tensor where rows will correspond to specific platform users, columns will correspond to individual artists and the cell contents will contain the number of hits a particular artist received from a particular user;</li>
#     <li style="text-align: justify">'<b><em>preferred_recommended</em></b>' will initially identify and print the top 'x' artists most reverenced by a specific user and, subsequently, print the top 'x' music artists most recommended to this particular user, excluding those who may be already included in the reverenced list (i.e. new recommendations only).</li>
# </ul>

# In[ ]:


def convert(f_data, f_nr_observations, f_nr_entities):
        """
        Generates (from a numpy array) a list of lists containing the number of hits per user (rows), per entity (columns).
        Each of the constituent lists will correspond to an observation / user (row).
        Each observation list will contain the number of hits (columns), one for each hit entity
        f_data - Input table (numpy array)
        f_nr_observations - Number of observations
        f_nr_entities - Number of entities hit in each observation
        """
        f_converted_data = []
        for f_id_user in range(1, f_nr_observations + 1):
            f_id_entity = f_data[:,1][f_data[:,0] == f_id_user].astype(int)
            f_id_hits = f_data[:,2][f_data[:,0] == f_id_user]
            f_hits = np.zeros(f_nr_entities)
            f_hits[f_id_entity - 1] = f_id_hits
            f_converted_data.append(list(f_hits))
        return f_converted_data


# In[ ]:


def preferred_recommended(f_artist_list, f_train_set, f_test_set, f_model, f_user_id, f_top=10):
        """
        Generates music artist recommendations for a particular platform user. 
        f_artist_list - List of artists and corresponding IDs
        f_train_set - Tensor containing training observations
        f_test_set - Tensor containing testing observations
        f_model - A RBM machine learning model previously instantiated
        f_user_id - The user for which preferred artists will be assessed and recommendations will be provided
        f_top - Number of most preferred and most recommended music artists for user 'f_user_id'
        """
        if f_user_id < 1515:
            f_user_sample = f_train_set[f_user_id - 1:f_user_id]
        else:
            f_user_sample = f_test_set[f_user_id - 1:f_user_id]
        f_prediction = f_model.predict(f_user_sample).numpy()
        f_user_sample = f_user_sample.numpy()
        f_user_sample = pd.Series(f_user_sample[0])
        f_user_sample = f_user_sample.sort_values(ascending=False)
        f_user_sample = f_user_sample.iloc[:f_top]
        f_fan_list = f_user_sample.index.values.tolist()
        print(f'\nUser {f_user_id} is a fan of...\n')
        for f_artist_id in f_fan_list:
            print(f_artist_list[f_artist_list.artist_id == f_artist_id + 1].iloc[0][1])
        f_prediction = pd.Series(f_prediction[0])
        f_prediction = f_prediction.sort_values(ascending=False)
        f_prediction_list = f_prediction.index.values.tolist()
        print(f'\nUser {f_user_id} may be interested in...\n')
        f_nb_recommendations = 0
        f_i = 0
        while f_nb_recommendations < f_top:
            f_pred_artist = f_prediction_list[f_i]
            if f_pred_artist not in f_fan_list:
                print(f_artist_list[f_artist_list.artist_id == f_pred_artist + 1].iloc[0][1])
                f_nb_recommendations += 1
            f_i += 1


# ## 4. The dataset

# <p style="text-align: justify">The dataset utilized in this deep learning exercise is a summarized, sanitized subset of the one released at <b>The 2nd International Workshop on Information Heterogeneity and Fusion in Recommender Systems</b> (HetRec 2011), currently hosted at the GroupLens website (<a href="https://grouplens.org/datasets/hetrec-2011/">here</a>).</p>
# <p style="text-align: justify">Sanitization included: (a) artist name mispelling correction and standardization; (b) reassignment of artists referenced with two or more artist id's; (c) removal of artists listed as 'unknown' or through their website addresses.</p>
# <p style="text-align: justify">The original dataset contains a larger number of files, including tag-related information, in addition to users, artists and scrobble counts. last.fm was contacted by the author and asked for some recent version of this content, in similar format, with no return until June 15th, 2020.</p>
# <p style="text-align: justify">Two dataset files were selected and preprocessed for use in this work:</p>
# <ol>
#     <li>'<b>lastfm_user_scrobbles.csv</b>' contains 92,792 scrobble counts ('scrobbles') for 17,493 artists ('artist_id') generated by 1,892 users ('user_id');</li>
#     <li>'<b>lastfm_artist_list.csv</b>' contains the list of 17,493 artists, referenced by an unique id code ('artist_id'), the same used in the first file.</li>
# </ol>

# In[ ]:


scrobbles = pd.read_csv('../input/lastfm-music-artist-scrobbles/lastfm_user_scrobbles.csv', header = 0)
scrobbles.head()


# <p style="text-align: justify">As anticipated in Section 2.1, scrobble counts are scaled on a per user basis with the code below. As some null values may be produced in the case of users who have generated only one scrobble for a given artist (maximum and minimum are the same, leading to a division by zero), the final scaled number of scrobbles of 0.5 is assigned to those users.</p>

# In[ ]:


scrobbles['scrobbles'] = scrobbles.groupby('user_id')[['scrobbles']].apply(lambda x: (x-x.min())/(x.max()-x.min()))
scrobbles['scrobbles'] = scrobbles['scrobbles'].fillna(0.5)
scrobbles.head()


# ## 5. Generating 'user versus ratings' tensors for training and testing

# <p style="text-align: justify">The scrobbles dataset is originally sorted based on ascending user ids. As generating recommendations for specific users is the ultimate objective of this exercise, it is necessary to maintain user scrobbles grouped. In addition, as roughly 20% of user scrobbles wil be segregated in a test set:</p>
# <ul>
#     <li>The training set will include the first 74,254 scrobbles, corresponding to users with 'user_id' ranging from 1 to 1,514;</li>
#     <li>The test set will include the remaining 18,538 scrobbles, corresponding to users with 'user_id' ranging from 1,515 to 1,892.</li>
# </ul>
# <p style="text-align: justify">A more sophisticated approach, with a random selection of user groups for the training and test sets, as well as a dynamic segregation of training users allowing for some cross-validation training, would add additional complexity and were not considered in this first release.</p>

# In[ ]:


training_size = 74254
training_set = scrobbles.iloc[:training_size, :]  # Until userID = 1514
test_set = scrobbles.iloc[training_size:, :]      # Starting at userID = 1515

training_set = training_set.values
test_set = test_set.values

training_set.shape, test_set.shape


# In[ ]:


nr_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nr_artists = int(max(max(training_set[:,1]), max(test_set[:,1])))


# In[ ]:


nr_users, nr_artists


# <p style="text-align: justify">At this point, both training and test sets are subsets of the original dataset, converted into numpy arrays. However, the model will be fed with a rearranged version of these tables, in which users will correspond to rows, artists to columns and the content of each cell will include the number of scrobbles generated by each user, for each artist.</p>

# In[ ]:


training_set = convert(training_set, nr_users, nr_artists)
test_set = convert(test_set, nr_users, nr_artists)


# <p style="text-align: justify">At last, both sets are converted into PyTorch float tensors:</p>

# In[ ]:


training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# ## 6. Deep learning

# ### 6.1. Creating the RBM as an instance of a Python object

# <p style="text-align: justify">The RBM is defined as having:</p>
# <ul>
#     <li>a number of visible nodes corresponding to the number of music artists (1,793) - one visible node per artist;</li>
#     <li>a number of hidden nodes arbitrarily defined and tuned;</li>
# </ul>
# <p style="text-align: justify">Comments on hyperparameter selection, tuning and implications are provided in Section 5.</p>

# In[ ]:


nv = len(training_set[0])
nh = 100
batch_size = 1
epoch = 50
metric = 'MAbsE'

model = RestrictedBoltzmannMachine(nv, nh)


# ### 6.2. Training & testing the RBM

# In[ ]:


model.train(nr_users, epoch, batch_size, training_set, metric)
model.test(nr_users, training_set, test_set, metric)


# ### 6.3. Providing recommendations

# <p style="text-align: justify">In addition to the traditional error metric-based performance assessment, it is of absolute importance to test the model through the generation of real recommendations for specific users. In order to do it wisely, two different users with evidently different music preferences were identified in the test set:</p>
# <ul>
#     <li>user_id # 1515 seems to be a fan of pop music and female muses in particular;</li>
#     <li>user_id # 1789 seems to prefer progressive and heavy metal rock artists.</li>
# </ul>
# <p style="text-align: justify">Recommendations are generated for both. The code below lists the 10 most 'scrobbled' music artists for each of these users, followed by the 10 most recommended artists in each case. Results are discussed in Section 5.</p>

# In[ ]:


artist_list = pd.read_csv('../input/lastfm-music-artist-scrobbles/lastfm_artist_list.csv', header = 0)


# <img src="https://i.imgur.com/vJKuVwM.png" width="500" height="100">

# In[ ]:


preferred_recommended(artist_list, training_set, test_set, model, 1515, 10)


# <img src="https://i.imgur.com/E9cuCbo.png" width="500" height="100">

# In[ ]:


preferred_recommended(artist_list, training_set, test_set, model, 1789, 10)


# ## 7. Discussion and final remarks

# <p style="text-align: justify">The Restricted Boltzmann Machine developed in this unsupervised learning exercise performed quite well from both the objective, error metric-based and the subjective, recommendation quality-based perspectives.</p>
# <p style="text-align: justify">Some initial considerations on hyperparameters:</p>
# <ul>
#     <li>Model variations with varied numbers of hidden nodes (25, 50, 100, 200, 500) were tested. Results were satisfactory (i.e. stable minimum losses and recommendations aligned with user profiles) with a minimum of 100 hidden nodes. No significant improvement was verified with larger numbers of hidden nodes;</li>
#     <li>The model accommodates observation batching for training. However, it has been noted over several simulation rounds that more accurate recommendations were obtained at the end with a batch size of 1;</li>
#     <li>Error metrics (Mean Absolute Error, or 'MAbsE') stabilize after 30 to 40 training epochs. A final number of 50 training epochs proved sufficient and was considered in the final release. </li>
# </ul>
# <p style="text-align: justify">Recommendations for the selected users were pretty much aligned with their most evident preferences. It shall though be noted that:</p>
# <ul>
#     <li>The lists of preferred and recommended artists displayed include only the top 10 in each case. However, these lists are long for some users, case in which artists not displayed, but present in the preferred artist list, certainly have a weight on final recommendations;</li>
#     <li>The scrobble count scaling strategy described in Sections 2.1 and 4 proved effective. Simulations were performed without it, and although error metrics converged as expected, the final recommendations were very much biased with a clear predominance of only the most popular artists in the artist universe.</li>
# </ul>

# In[ ]:


end_time = datetime.now()
print(f'Time elapsed: {end_time - start_time}')


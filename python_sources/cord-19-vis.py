from gensim import matutils

import numpy as np
import pandas as pd

# global var
dictionary = None


"""
Selecting the number of clusters with silhouette analysis on KMeans clustering
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
"""
def plot_tm_clusters(tm_ranker, ax2, ax1, set_y_label=True,
                     n_clusters = 20):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import matplotlib.cm as mcm
    from sklearn.metrics import silhouette_samples, silhouette_score

    colors = mcm.tab20(np.linspace(0, 1, n_clusters))
    
    X = tm_ranker.D

    k_means = KMeans(n_clusters=n_clusters, random_state=1)
    cluster_labels = k_means.fit_predict(X) 

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=0, random_state=0, angle=.99, init='pca')
    tsne = tsne_model.fit_transform(X)
    
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.2, 0.7]
    ax1.set_xlim([-0.15, 0.7])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        #color = mcm.nipy_spectral(float(i) / n_clusters)
        color = colors[i]
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    #ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    if set_y_label:
        ax1.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.15, 0, 0.1, 0.2, 0.4, 0.5, 0.7])
    
    ax2.scatter(tsne[:,0], tsne[:,1], c=colors[cluster_labels],
                marker='.', s=30)
    ax2.set_xlabel('1st component')
    if set_y_label:
        ax2.set_ylabel('2nd component')



"""
Functions inspired from:
    Topic modeling visualization – How to present the results of LDA models?
    https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
The code is close to the code in the blog, but now that we understand the process, we can rewrite the functions
to be more simple/packed and vectorized.
"""
def get_top_words(model, corpus):
    # Sentence Coloring of N Sentences
    def topics_per_document(model, corpus):
        dominant_topics = []
        topic_percentages = []
        for i, doc in enumerate(corpus):
            topic_percs = model.project(doc)[0]
            dominant_topic = np.argmax(topic_percs)
            dominant_topics.append((i, dominant_topic))
            x = list( zip(np.arange(len(topic_percs)), topic_percs) )
            topic_percentages.append( x )
        return(dominant_topics, topic_percentages)

    dominant_topics, topic_percentages = topics_per_document(model, corpus)

    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'topic_id'])
    dominant_topic_in_each_doc = df.groupby('topic_id').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='num_docs').reset_index()

    # Total Topic Distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = (topic_weightage_by_doc.mean().to_frame(name='prob').reset_index()
                                 .rename(columns={'index': 'topic_id'}))

    # Top(num_words) Keywords for each Topic
    def get_topic_top_words(model, num_words=5):
        # Like gensim:ldamodel:show_topics
        shown = []
        for i, topic in enumerate(model.B):
            bestn = matutils.argsort(topic, num_words, reverse=True)
            topic = [(i, dictionary.id2token[id]) for id in bestn]
            shown.extend(topic)

        return shown

    topic_topwords = get_topic_top_words(model)

    df_topwords_stacked = pd.DataFrame(topic_topwords, columns=['topic_id', 'words'])
    df_topwords = df_topwords_stacked.groupby('topic_id').agg(', \n'.join)
    df_topwords.reset_index(level=0,inplace=True)

    df_topwords = df_topwords.merge(df_dominant_topic_in_each_doc, on=['topic_id'], how='inner')
    df_topwords = df_topwords.merge(df_topic_weightage_by_doc, on=['topic_id'], how='inner')

    df_topwords['words'] = 'Topic ' + df_topwords.topic_id.astype(str) + '\n' + df_topwords.words.astype(str)
    
    return df_topwords

def plot_topics_dist(ranker, corpus, ax1, ax2, title, set_y_label=True):
    df_topwords = get_top_words(ranker, corpus)
    
    df_top5 = df_topwords.sort_values(by=['num_docs'], ascending=False).iloc[:5]

    # Topic Distribution by Dominant Topics
    ax1.bar(x='words', height='num_docs', data=df_top5,
            width=.5, color='firebrick')
    ax1.set_title(title, fontdict=dict(size=10))
    if set_y_label:
        ax1.set_ylabel('Number of Documents')

    # Topic Distribution by Topic mean prob
    df_top5 = df_topwords.sort_values(by=['prob'], ascending=False).iloc[:5]
    ax2.bar(x='words', height='prob', data=df_top5,
            width=.5, color='steelblue')
    ax2.set_title(title, fontdict=dict(size=10))
    if set_y_label:
        ax2.set_ylabel('Mean probability')


"""
Porting pyLDAvis to work with our models (LDI,NMF)
"""

import pyLDAvis.gensim
import funcy as fp

"""
https://github.com/bmabey/pyLDAvis/blob/2da07084e9df7d51a0daf240db1a64022e3023a5/pyLDAvis/gensim.py#L14
"""
def extract_data(topic_model, corpus, dictionary, top_topics = None):
    vocab = list(dictionary.token2id.keys())
    beta = 0.01
    fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)
    term_freqs = corpus.TF.sum(axis=0).ravel()[fnames_argsort]
    term_freqs[term_freqs == 0] = beta
    doc_lengths = corpus.TF.sum(axis=1).ravel()

    assert term_freqs.shape[0] == len(dictionary), \
        'Term frequencies and dictionary have different shape {} != {}'.format(
        term_freqs.shape[0], len(dictionary))
    assert doc_lengths.shape[0] == len(corpus), \
        'Document lengths and corpus have different sizes {} != {}'.format(
        doc_lengths.shape[0], len(corpus))

    num_topics = topic_model.D.shape[1]
    
    if not top_topics is None:
        doc_topic_dists = topic_model.D[:, top_topics].copy()
        doc_topic_dists[doc_topic_dists.sum(axis=1)==0] = 1/num_topics
        topic_term_dists = topic_model.B[top_topics][:, fnames_argsort].copy()
    else:
        doc_topic_dists = topic_model.D.copy()
        topic_term_dists = topic_model.B[:, fnames_argsort].copy()
    
    # Be sure we don't have zero term topic prob. Why not?
    # This is only for NMF model
    topic_term_dists[topic_term_dists == 0] = 1e-18
    
    #gamma.sum(axis=1)[:, None]
    doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)[:, None]
    
    

    assert topic_term_dists.shape[0] == doc_topic_dists.shape[1]

    return {'topic_term_dists': topic_term_dists, 'doc_topic_dists': doc_topic_dists,
            'doc_lengths': doc_lengths, 'vocab': vocab, 'term_frequency': term_freqs}

def prepare(topic_model, corpus, dictionary, top_topics = None, **kwargs):
    opts = fp.merge(extract_data(topic_model, corpus, dictionary, top_topics), kwargs)
    return pyLDAvis.prepare(**opts)


"""
pyLDAvis now is incompatible with Kaggle kernels.
The reason is pyLDAvis need a fixed width 1350px to draw graphs,
but the kernel is shown into a frame usully in smaller size.
All the code bellow do just two things:
1) in CSS set the width 1350px to the graph and no more to the container;
2) `visid` is now mandatory.

WARNING: You are responsable to provide different name (visid) to each graph
"""
from pyLDAvis import urls
import jinja2
import re
import json

def prepared_data_to_html(data, d3_url=None, ldavis_url=None, ldavis_css_url=None,
                          template_type="general", visid=None, use_http=False):

    assert not visid is None, "We should provide visid (div id)"
    
    template = jinja2.Template("""

<style>
path {
  fill: none;
  stroke: none;
}

.xaxis .tick.major {
    fill: black;
    stroke: black;
    stroke-width: 0.1;
    opacity: 0.7;
}

.slideraxis {
    fill: black;
    stroke: black;
    stroke-width: 0.4;
    opacity: 1;
}

text {
    font-family: sans-serif;
    font-size: 11px;
}

#{{ visid_raw }} {
    width:1350px;
}
</style>

<div id={{ visid }}></div>
<script type="text/javascript">

    var {{ visid_raw }}_data = {{ vis_json }};

    require.config({paths: {d3: "{{ d3_url[:-3] }}"}});
    require(["d3"], function(d3){
        window.d3 = d3;
        $.getScript("{{ ldavis_url }}", function(){
            new LDAvis("#" + {{ visid }}, {{ visid_raw }}_data);
        });
    });

</script>
""")

    d3_url = urls.D3_URL
    ldavis_url = urls.LDAVIS_URL
    
    if re.search(' ', visid):
        raise ValueError("visid must not contain spaces")

    return template.render(visid=json.dumps(visid),
                           visid_raw=visid,
                           d3_url=d3_url,
                           ldavis_url=ldavis_url,
                           vis_json=data.to_json())

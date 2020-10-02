#!/usr/bin/env python
# coding: utf-8

# # Predicting Specialty with Prescribing Behavior
# 
# In this analysis, we explore the task of predicting a medical provider's specialty given their prescribing behavior with a focus on understanding the structure of the data. The distribution of specialties in this dataset is highly imbalanced, making this an interesting exercise in multiclass classification. In addition to this predictive modelling, I perform exploratory analysis using Latent Dirichlet Allocation (LDA).
# 
# The data, compiled and released by Roam Analytics, focuses on Medicare Part D claims and consists of providers, information such as region and gender, and counts of drugs that those providers prescribed. The [original blog post](https://roamanalytics.com/2016/09/13/prescription-based-prediction/) introduces the data in more detail and summarizes the motivation for the work as follows:
# 
# > We expect a doctor's prescribing behavior to be governed by many complex, interacting factors related to her specialty, the place she works, her personal preferences, and so forth. How much of this information is hidden away in the Medicare Part D database?

# <a></a>

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.offline as ply
import plotly.graph_objs as go
import seaborn as sns
import sklearn.metrics.base
import warnings
from collections import Counter
from pandas.io.json import json_normalize
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, train_test_split,
                                     cross_val_predict)
ply.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Preparation
# 
# The data is provided in JSONL format, so some light preprocessing is necessary to get this into a usable format for analysis.
# 
# We perform the same row filtering technique as that described by the original authors, namely ignoring rows corresponding to specialties with fewer than 50 records and those for which there are fewer than 50 unique drugs prescribed.

# In[2]:


unfiltered_data = pd.read_json('../input/roam_prescription_based_prediction.jsonl', lines=True)


# In[3]:


# Filter out rows for providers with <50 unique prescribed drugs. Then, remove providers that
# correspond to specialties with <50 providers in the filtered dataset.
data = unfiltered_data[unfiltered_data.cms_prescription_counts.apply(lambda x: len(x.keys())) >= 50]
specialty_counts = Counter(data.provider_variables.apply(lambda x: x['specialty']))
specialties_to_ignore = set(
    specialty for specialty, _ in filter(lambda x: x[1] < 50, specialty_counts.items()))
data = data[data.provider_variables.apply(lambda x: x['specialty'] not in specialties_to_ignore)]
data.head()


# In[4]:


# Expand the provider variables into a data frame.
provider_variables = json_normalize(data=data.provider_variables)
data.drop('provider_variables', axis=1, inplace=True)
provider_variables.head()


# In[5]:


# Transform prescription count column into sparse matrix.
vectorizer = DictVectorizer(sparse=True)
X = vectorizer.fit_transform(data.cms_prescription_counts)


# ## Exploratory Analysis
# 
# There are 29 unique classes across the response and the distribution is highly imbalanced as mentioned previously. Cardiovascular Disease accounts for nearly 20% of all of the samples, while the 12 lowest frequency classes account for fewer than 1% each. This will be important to keep in mind later to ensure that the model doesn't bias too heavily towards the high frequency classes.

# In[6]:


provider_variables.specialty.unique().shape[0]


# In[7]:


plt.figure(figsize=(10,10))
counts = pd.Series(provider_variables.specialty).value_counts()[::-1] 
plt.barh(counts.index, counts.values)
for i, v in enumerate(counts.values):
    plt.text(v, i, '{:.1f}%'.format(100 * v / provider_variables.specialty.shape[0]))


# Also, it appears that there are some very similar classes that could be tricky for the model to distinguish. For example, Child & Adolescent Psychiatry versus Psychiatry, and Geriatric Medicine versus Gerontology. We'll use PCA to explore these relationships further as well as get an idea of the expected separation between classes.

# In[8]:


# Sum the counts across all providers for each specialty and normalize.
counts_per_specialty = pd.DataFrame(
    X.todense(), index=provider_variables.specialty
).groupby(
    'specialty'
).sum()
dist_per_specialty = counts_per_specialty.values /     counts_per_specialty.values.sum(axis=1).reshape((-1, 1))


# In[9]:


# Fit PCA and keep only 2 dimensions for plotting.
pca = PCA(n_components=2)
specialties_pca = pca.fit_transform(dist_per_specialty)


# After plotting the first two principal components, there is clear separation for a few specialized classes such as Neurology and Rheumatology. This indicates that these classes will likely be among the easiest to predict accurately.
# 
# There are also clusters of similar classes forming in this plot. Psychiatry, Child & Adolescent Psychiatry, and Psych/Mental Health is one example. Medical Oncology and Hematology & Oncology are also close in this space, as are Pain Medicine and Interventional Pain Medicine. The plot is interactive and can be cropped to zoom into regions with dense clusters.

# In[10]:


# Plot the first two principal components.
scatter = go.Scattergl(
    x=specialties_pca[:,0],
    y=specialties_pca[:,1],
    mode='markers',
    text=counts_per_specialty.index,
    hoverinfo='text',
    marker=dict(
        opacity=0.5
    )
)
layout = go.Layout(
    title='Decomposition of Empirical Drug Distributions',
    xaxis=dict(
        title='PC1'
    ),
    yaxis=dict(
        title='PC2'
    ),
    hovermode='closest'
)
figure = go.Figure(data=[scatter], layout=layout)
ply.iplot(figure)


# The features we'll use to predict specialty are the provder's prescription counts. As cited in the blog post, prescription counts follow an approximate Zipfian distribution meaning that the frequency each drug is prescribed is inversely proportional to that drug's frequency rank order. This becomes evident when plotting the count of each prescription against its rank in our filtered dataset.

# In[11]:


top_n = 500
values = np.array(X.sum(axis=0))[0]
sort_order = np.argsort(values)[::-1][:top_n]
plt.figure(figsize=(10,5))
plt.bar(range(top_n), values[sort_order], width=1)
plt.xlabel('Rank')
plt.ylabel('Prescription Count')
plt.title('Drug Frequency Distribution (top {})'.format(top_n))


# Taking a look at the top prescribed drugs, we see several related to cardiovascular health. Lisinopril and Amlodipine are used to treat high blood pressure, and Simvastatin is a cholesterol medication. Given that the top specialty in this dataset is Cardiovascular Disease, this list is not all that surprising.
# 
# The top 15 drugs account for nearly a third of all prescriptions in this data. However, it's important to note that for each provider, the data only contains counts for the drugs that they prescribed more than 10 times in that year (presumably for privacy/anonymity reasons?). I guess that there is an even longer tail that we're missing here because of that.

# In[12]:


top_n_drugs = 15
top_drugs_df = pd.DataFrame({
    'Drug Name': np.array(vectorizer.feature_names_)[sort_order[:top_n_drugs]],
    'Count': list(map(int, values[sort_order[:top_n_drugs]]))
})
top_drugs_df['Proportion of All Prescriptions'] = top_drugs_df['Count'] / values.sum()
top_drugs_df['Cumulative Proportion of All Prescriptions'] =     top_drugs_df['Proportion of All Prescriptions'].cumsum()
top_drugs_df


# Next, we will look at correlation between the drug features. Given that we have generics and brand name drugs in this dataset, we expect there to be some correlation.
# 
# There appear to be several drugs prescribed by only one provider in the dataset which causes perfect correlation between some features. Looking at the distribution of unique providers for each drug, we see that there are quite a few drugs that have a limited number of providers. We may want to remove some of these as predictors.
# 
# Looking at the correlated pairs excluding drugs prescribed by only one provider, we see a lot of HIV drugs.

# In[54]:


drug_correlations = np.corrcoef(X.todense(), rowvar=False)
np.fill_diagonal(drug_correlations, 0)
drug_correlations = np.triu(drug_correlations)


# In[105]:


unique_provider_counts = np.array((X > 0).sum(axis=0))[0]
plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
pd.Series(unique_provider_counts).plot(
    kind='hist', title='Drugs by Number of Unique Providers')
plt.subplot(1, 2, 2)
clip_at = 50
pd.Series(list(filter(lambda x: x < clip_at, unique_provider_counts))).plot(
    kind='hist', title='Drugs by Number of Unique Providers (clipped at {})'.format(clip_at))


# In[113]:


correlated_pairs = zip(*(drug_correlations > .85).nonzero())
pd.DataFrame(
    sorted([
        (vectorizer.feature_names_[i], vectorizer.feature_names_[j], drug_correlations[i, j],
         unique_provider_counts[i], unique_provider_counts[j])
        for i, j in correlated_pairs
        if unique_provider_counts[i] > 1 and unique_provider_counts[j] > 1],
        key=lambda x: x[2], reverse=True),
    columns=['Drug 1', 'Drug 2', 'Correlation', 'Drug 1 Provider Count', 'Drug 2 Provider Count']
)


# When looking at word counts in a natural language corpus, a similar power law pattern to the one seen above in the drug frequency emerges. A common exploratory technique used in the analysis of language is topic modelling, which refers to the unsupervised identification of topics, or groups of related words, in text. Latent Dirichlet Allocation (LDA) is one such model.
# 
# In this dataset, we can think of each provider as a document and each drug that that provider prescribes as a word in the document. Using this analogy, we can apply LDA to identify groups of related drugs. LDA also gives vector representations for documents which lets us use it for dimensionality reduction purposes as well. This type of analysis can help denoise the features we send into a predictive model (especially when it comes to brand name drugs versus generics).
# 
# LDA requires setting a number of topics a priori, and it's a bit of an art to find the best value. We fix it at 50, but it could be interesting to experiment with this number further.

# In[ ]:


# Fit LDA model fixing k at 50.
lda = LatentDirichletAllocation(n_components=50, random_state=523)
lda.fit(X)


# Each learned topic from LDA is represented as a probability distribution over the unique drugs. One way to interpret the learned topics is to simply look at the highest probability drugs for each one. Considering there are a small number of very popular drugs as we saw before, this interpretation can be problematic because these popular drugs could have a high probability relative to other drugs in the topic but a low probability relative to their frequency across the entire dataset.
# 
# What we really care about is drugs that are not only highly probable in the topic but also have an anomalously high probability relative to their frequency across the entire dataset. There are a number of reweighting schemes to achieve this effect, but we implement the one described [here](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf). In short, the relevance score $r(w, k | \lambda)$ for drug $w$ in topic $k$ can be computed as:
# 
# $r(w, k | \lambda) = \lambda * \log(\phi_{kw}) + (1 - \lambda) * \log(\frac{\phi_{kw}}{p_w})$
# 
# where $\phi_{kw}$ is the learned probability of drug $w$ in topic $k$ and $p_w$ is the empirical probability of drug $w$ across the entire dataset. $\lambda$ is a parameter that controls the weighting between the two competing terms. The authors of the paper linked above found an optimal value of about .6, so that's what we'll use here.

# In[ ]:


p_w = np.array(X.sum(axis=0))[0] / X.sum()

def relevance(phi_kw, lambda_=.6):
    """Given a topic vector (probability of each drug given the topic) return a
    relevance score vector."""
    return lambda_ * np.log(phi_kw) + (1 - lambda_) * np.log (phi_kw / p_w)


# In[ ]:


# Extract the distribution over drugs for each topic, apply the relevance weighting above,
# and show the top N drugs for each topic.
def lda_summary_df(lda, top_n_drugs=3, **kwargs):
    topic_term_distribution = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    rows = []
    for i in range(lda.n_components):
        top_term_indices = np.argsort(
            relevance(topic_term_distribution[i], **kwargs))[::-1][:top_n_drugs]
        rows.append(np.array(vectorizer.feature_names_)[top_term_indices])
    top_drugs_per_topic = pd.DataFrame(rows)
    top_drugs_per_topic.index = map(lambda x: 'Topic {}'.format(x), range(lda.n_components))
    top_drugs_per_topic.columns = map(lambda x: 'Drug #{}'.format(x+1), range(top_n_drugs))
    return top_drugs_per_topic


# Inspecting the learned topics from LDA, we see patterns of related drugs emerging. For example:
# 
# * Topic 4 (TEMAZEPAM, LORAZEPAM, CLONAZEPAM): Sedatives.
# * Topic 7 (COPAXONE, BACLOFEN, AVONEX): MS.
# * Topic 10 (NORVIR, TRUVADA, ISENTRESS): HIV medications.
# * Topic 13 (MORPHINE SULFATE ER, OXYCODONE HCL, OXYCODONE HCL-ACETAMINOPHEN): Opiate pain relievers.
# * Topic 28 (JANUMET, TRADJENTA, JANUVIA): Diabetes.
# * Topic 47 (LEVETIRACETAM, VIMPAT, LAMOTRIGINE): Used to treat seizures.

# In[ ]:


lda_summary = lda_summary_df(lda)
lda_summary


# In[ ]:


# Show the most prevalent topics found in the dataset measured by the mean probability
# assigned to that topic across all rows.
provider_topic_matrix = lda.transform(X)
topic_prevalence = provider_topic_matrix.mean(axis=0)
topic_order = np.argsort(topic_prevalence)[::-1]
lda_topic_prevalence = pd.DataFrame(
    dict(Topic=topic_order, Prevalence=topic_prevalence[topic_order]))
lda_topic_prevalence['Top 3 Drugs from Topic'] =     lda_topic_prevalence.Topic.apply(lambda x: ', '.join(lda_summary.iloc[x]))
lda_topic_prevalence


# ## Predicting Provider Specialty
# 
# Next, we'll start building some classification models. Through error analysis and careful interpretation of metrics, we'll keep a close eye on the problem of imbalanced classes.
# 
# The F1 score is a standard metric for evaluating classification models, and there are a few ways to generalize this metric to the multiclass case. We will primarily use the macro averaging strategy which refers to averaging the F1 score computed individually for each class. Note that this means that classes are weighted equally in the scoring metric despite their widely varying sizes. The micro averaging strategy refers to summing up TP, FP, etc counts across all the classes and computing F1 using those numbers.
# 
# Because there are some low count classes, it is possible that some of the trained models will not produce any predictions for a given class. In this case, precision is undefined for that class which in turn means that the macro F1 is undefined. In these cases, the default behavior is to fill with a score of 0. Lots of warnings are thrown when this happens, but since this is expected, we'll turn them off.

# In[127]:


warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.UndefinedMetricWarning)


# Finally, we split the data into a training set and a test set which will be used to evaluate errors. We'll also use a stratified 3-fold cross validation on the training set to tune hyperparameters, obtain estimates of test metrics, and perform error analysis. We initialize the splitter class below to be able to use the same folds across the various models explored.

# In[117]:


X_train, X_test, X_train_lda, X_test_lda, provider_variables_train, provider_variables_test =    train_test_split(X, lda.transform(X), provider_variables,
                     stratify=provider_variables.specialty, test_size=.25)


# In[118]:


splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=237)


# ### Logistic Regression
# 
# We look at a couple different ways of applying logistic regression. The first is to use a TFIDF transform on the drug counts before feeding the transformed result into the model as features. This is a standard practice in NLP and is used to better highlight the drugs that are prescribed unusually often by a given provider. The second is to use the low dimensional vector representations generated by LDA as features. These vectors can be interpreted as distributions over the latent topics, so this essentially turns our feature space from individual drugs into higher level groups such as sedatives, HIV meds, diabetes meds, etc.

# In[136]:


class_labels = provider_variables_train.specialty.value_counts()

def fit_lr_model(X, y, max_iter=100, param_grid={'C': np.logspace(0, 10, 10)}):
    """Fit a logistic regression model and obtain cross validated predictions
    for error analysis."""

    # Fit model.
    model = GridSearchCV(
        LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=max_iter
        ),
        scoring=['f1_macro', 'f1_micro'],
        param_grid=param_grid,
        cv=splitter,
        return_train_score=True,
        refit='f1_macro'
    )
    model.fit(X, y)
    
    # Obtain cross validated predictions.
    cv_preds = cross_val_predict(model.best_estimator_, X, y, cv=splitter)
    
    # Summarize F1 score on the cross validated predictions.
    f1_scores = f1_score(y, cv_preds, labels=class_labels.index, average=None)
    class_summary = pd.DataFrame(
        list(zip(class_labels.index, f1_scores, class_labels.values, np.argsort(-f1_scores).argsort() + 1)),
        columns=['Specialty', 'F1', 'True Count', 'F1 Rank']
    )
    
    # Confusion matrix on cross validated predictions.
    confusion_matrix_cv_preds = pd.DataFrame(
        confusion_matrix(y, cv_preds, labels=class_labels.index))
    confusion_matrix_cv_preds.index = class_labels.index
    confusion_matrix_cv_preds.columns = class_labels.index
    return model, cv_preds, class_summary, confusion_matrix_cv_preds


# #### TFIDF Features

# In[120]:


# Obtain a TFIDF transform and apply to both the train and test sets.
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train)
X_train_tfidf = tfidf_transformer.transform(X_train)
X_test_tfidf = tfidf_transformer.transform(X_test)


# In[137]:


# Train a logistic regression model searching over a range of regularization strengths.
tfidf_model, cv_preds_tfidf, class_summary_tfidf, confusion_matrix_tfidf = fit_lr_model(
    X_train_tfidf, provider_variables_train.specialty)


# Notice all of the convergence warnings. Increasing the number of iterations had no meaningful impact on loss and scoring metrics. This requires further investigation.

# In[130]:


tfidf_model.cv_results_


# To analyze errors, we look at the F1 score breakdown by class and confusion matrix of predictions. While the performace is definitely better for larger classes, the linear relationship between the class rank by size and the class rank by F1 has a very low $R^2$. The top performing classes are Rheumatology, Neurology, and Nephrology which agrees with some of the findings from the PCA analysis earlier, and the worst performing are Child & Adolescent Psychiatry, Adolescent Medicine, and Acute Care. Highly specialized fields appear to perform best, while those that are highly related to other fields (e.g. the adolescent analogs) or very broad (e.g. Primary Care) perform worst.

# In[138]:


# Summary of F1 scores across classes on the cross validated predictions.
class_summary_tfidf


# In[139]:


plt.figure(figsize=(10, 10))
plt.scatter(class_summary_tfidf.index, class_summary_tfidf['F1 Rank'])
plt.xlabel('Class Rank (by size)')
plt.ylabel('Class Rank (by F1)')
plt.title('Relationship between Class Size and per class F1')


# Looking at the confusion matrix, we can definitely see patterns in the errors for the low performing classes. The biggest issue appears to be distinguishing between very similar classes rather than failing on strictly small sample classes. For example:
# * Child & Adolescent Psychiatry getting mistaken for Psychiatry.
# * Acute Care getting mistaken for Family.
# * Primary Care getting mistaken for Family.
# * Interventional Cardiology getting mistaken for Cardiovascular Disease.

# In[140]:


# Inspect the confustion matrix of predictions. The number in row i, column j is the count
# known to be in class i and predicted to be class j.
confusion_matrix_tfidf


# #### LDA Features

# In[ ]:


# Train a logistic regression model on LDA features. With such a comparatively low number of
# features, regularization really shouldn't be necessary here.
ldalr_model, cv_preds_lda, class_summary_lda, confusion_matrix_lda = fit_lr_model(
    X_train_lda, provider_variables_train.specialty, param_grid={'C': [10000000]})


# In[ ]:


ldalr_model.cv_results_


# In[ ]:


# Summary of F1 scores across classes on the cross validated predictions.
class_summary_lda


# In[ ]:


# Inspect the confustion matrix of predictions. The number in row i, column j is the count
# known to be in class i and predicted to be class j.
confusion_matrix_lda


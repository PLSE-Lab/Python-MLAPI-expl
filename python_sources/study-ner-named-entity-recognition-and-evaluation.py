#!/usr/bin/env python
# coding: utf-8

# # Source
# This example start from the article [Named-Entity evaluation metrics based on entity-level](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)

# In[ ]:


get_ipython().system('pip install sklearn-crfsuite')


# In[ ]:


import nltk
import sklearn_crfsuite

from copy import deepcopy
from collections import defaultdict

from sklearn_crfsuite import metrics


# # Added code
# This code is into a [git repo](https://github.com/davidsbatista/NER-Evaluation/blob/master/ner_evaluation/ner_eval.py)

# In[ ]:


from copy import deepcopy
from collections import namedtuple

Entity = namedtuple("Entity", "e_type start_offset end_offset")


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities):
    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}
    target_tags_no_schema = ['MISC', 'PER', 'LOC', 'ORG']

    # overall results
    evaluation = {'strict': deepcopy(eval_metrics),
                  'ent_type': deepcopy(eval_metrics),
                  'partial': deepcopy(eval_metrics),
                  'exact': deepcopy(eval_metrics)}

    # results by entity type
    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in target_tags_no_schema}

    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # Check each of the potential scenarios in turn. See 
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation. 

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
            evaluation['exact']['correct'] += 1
            evaluation['partial']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['exact']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['partial']['correct'] += 1

        else:

            # check for overlaps with any of the true entities

            for true in true_named_entities:

                pred_range = range(pred.start_offset, pred.end_offset)
                true_range = range(true.start_offset, true.end_offset)

                # Scenario IV: Offsets match, but entity type is wrong

                if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset                         and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    evaluation['partial']['correct'] += 1
                    evaluation['exact']['correct'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['partial']['correct'] += 1
                    evaluation_agg_entities_type[true.e_type]['exact']['correct'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # check for an overlap i.e. not exact boundary match, with true entities

                elif find_overlap(true_range, pred_range):

                    true_which_overlapped_with_pred.append(true)

                    # Scenario V: There is an overlap (but offsets do not match
                    # exactly), and the entity type is the same.
                    # 2.1 overlaps with the same entity type

                    if pred.e_type == true.e_type:

                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['correct'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        found_overlap = True
                        break

                    # Scenario VI: Entities overlap, but the entity type is 
                    # different.

                    else:
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        # Results against the true entity

                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        # Results against the predicted entity

                        # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                        found_overlap = True
                        break

            # Scenario II: Entities are spurious (i.e., over-generated).

            if not found_overlap:
                # overall results
                evaluation['strict']['spurious'] += 1
                evaluation['ent_type']['spurious'] += 1
                evaluation['partial']['spurious'] += 1
                evaluation['exact']['spurious'] += 1

                # aggregated by entity type results
                evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['ent_type']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['partial']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['exact']['spurious'] += 1

    # Scenario III: Entity was missed entirely.

    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1
            evaluation['partial']['missed'] += 1
            evaluation['exact']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['partial']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['exact']['missed'] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.

    for eval_type in evaluation:
        evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level 
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        for eval_type in entity_level:

            evaluation_agg_entities_type[entity_type][eval_type] = compute_actual_possible(
                entity_level[eval_type]
            )

    return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges
    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().
    Examples:
    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results['correct']
    incorrect = results['incorrect']
    partial = results['partial']
    missed = results['missed']
    spurious = results['spurious']

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def compute_precision_recall(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results['partial']
    correct = results['correct']

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall
    results["f1-score"] = 2*precision*recall/(precision + recall)

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {key: compute_precision_recall(value, True) for key, value in results.items() if
                 key in ['partial', 'ent_type']}
    results_b = {key: compute_precision_recall(value) for key, value in results.items() if
                 key in ['strict', 'exact']}

    results = {**results_a, **results_b}

    return results


# # Dataset

# In[ ]:


nltk.corpus.conll2002.fileids()
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


# In[ ]:


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# # Prepare dataset for CRF

# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = [sent2features(s) for s in train_sents]\ny_train = [sent2labels(s) for s in train_sents]\n\nX_test = [sent2features(s) for s in test_sents]\ny_test = [sent2labels(s) for s in test_sents]')


# # Train CRF

# In[ ]:


get_ipython().run_cell_magic('time', '', "crf = sklearn_crfsuite.CRF(\n    algorithm='lbfgs',\n    c1=0.1,\n    c2=0.1,\n    max_iterations=100,\n    all_possible_transitions=True\n)\ncrf.fit(X_train, y_train)")


# # Evaluate CRF

# In[ ]:


y_pred = crf.predict(X_test)


# In[ ]:


test_sents_labels = []
for sentence in test_sents:
    sentence = [token[2] for token in sentence]
    test_sents_labels.append(sentence)


# In[ ]:


index = 2
test_sents_labels[index]


# In[ ]:


true = collect_named_entities(test_sents_labels[index])
pred = collect_named_entities(y_pred[index])


# In[ ]:


true


# In[ ]:


pred


# In[ ]:


compute_metrics(true, pred)


# # Train with spacy

# In[ ]:


import spacy
from spacy.util import compounding, minibatch
import random


# In[ ]:


def create_model(labels):
    nlp = spacy.blank("it")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)
    for label in labels:
        ner.add_label(str(label))
    optimizer = nlp.begin_training()
    return nlp, optimizer

def train_model(n_iter, nlp, optimizer, train_data):
    # train_data = [(text, {entities:[(start_offset, end_offset, label), ]})]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           drop=0.35, losses=losses)
            print("(%d/%d)Losses" % ((itn+1), n_iter), losses)
            
def predict_model(nlp, pred_data):
    results = []
    for text, ent in pred_data:
        doc = nlp(text)
        results.append((
            text,
            {'entities':[
                [len(str(doc[0:x.start])) + 1, len(str(doc[0:x.start])) + len(str(doc[x.start:x.end])) + 1,
                     x.label_]
                    for x in doc.ents
            ]}
        ))
    return results


# # Prepare dataset for Spacy

# In[ ]:


def prepare_dataset(sents):
    dataset = []
    labels = set()
    for sent in sents:
        text = ' '
        entities = []
        for w in sent:
            text += w[0] + ' '
            label = w[2]
            if label != 'O':
                labels.add(label)
                entities.append([len(text)-len(w[0])-2,len(text)-2,label])
        text = text.strip()
        dataset.append( (text, {'entities': entities}) )
    return dataset, labels


# In[ ]:


train_dataset, labels = prepare_dataset(train_sents)
test_dataset, _ = prepare_dataset(test_sents)


# In[ ]:


ner, optimizer = create_model(labels)


# In[ ]:


epochs = 1
train_model(epochs ,ner, optimizer, train_dataset)


# # Evaluate Model

# In[ ]:


from spacy.gold import GoldParse
from spacy.scorer import Scorer
def evaluate_ner(ner_model, test_dataset):
    scorer = Scorer()
    for text, entities in test_dataset:
        doc_gold_text = ner_model.make_doc(text)
        gold = GoldParse(doc_gold_text, entities=entities['entities'])
        pred_value = ner_model(text)
        scorer.score(pred_value, gold)
    return scorer.scores


# In[ ]:


evaluate_ner(ner, test_dataset)


# In[ ]:


# compute_metrics(true, pred)


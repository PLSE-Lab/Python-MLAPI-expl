import pandas as pd
import numpy as np
import os
import logging
from collections import Counter
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functools import partial
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Iterator, Tuple, Iterable, List
import sys
import copy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(funcName)s:%(lineno)d [%(levelname)s] %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler('out.log')])
logger = logging.getLogger(__name__)


def handle_exception(exc_type, exc_value, exc_traceback):
    exc_info = (exc_type, exc_value, exc_traceback)
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(*exc_info)
        return
    logger.error("Uncaught exception", exc_info=exc_info)


sys.excepthook = handle_exception


def set_seeds(seed):
    """ set seed for numpy and pytorch """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)


def _get_candidate_dirs():
    """ if file candidate_directories.csv exists, read each line as a path
        return the set of paths read
    """
    candidate_dirs = set(['.', '../input'])
    if 'candidate_directories.csv' in os.listdir('.'):
        df = pd.read_csv('candidate_directories.csv', names=['directory'])
        candidate_dirs |= set(df['directory'].tolist())
    return candidate_dirs


def get_data_dir():
    """ helper to find directory containing input directory depending
        on machine being used
    """
    candidate_dirs = _get_candidate_dirs()
    data_files = set(['train.csv', 'test.csv'])
    for candidate_dir in candidate_dirs:
        candidate_dir = os.path.expanduser(candidate_dir)
        if not os.path.exists(candidate_dir):
            logger.debug(f"could not find directory {candidate_dir}")
            continue
        dir_files = set(os.listdir(candidate_dir))
        if data_files < dir_files:
            return candidate_dir
    raise FileNotFoundError(f"{data_files} not found in {candidate_dirs}")


def get_log_dir():
    """ return log dir, create it if does not exist """
    log_dir = './logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return log_dir


class QuoraInsincereDataset(Dataset):

    def __init__(self, csv_path, lower):
        self.dataset = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def load_data(data_dir, use_saved=False, shuffle=True):
    """ load training dataset and test dataset """
    saved_files = set(['train.pkl', 'test.pkl'])
    if use_saved and saved_files < set(os.listdir(data_dir)):
        train_data = pd.read_pickle(os.path.join(data_dir, 'train.pkl'))
        test_data = pd.read_pickle(os.path.join(data_dir, 'test.pkl'))
    else:
        train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    if shuffle:
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        test_data = test_data.sample(frac=1).reset_index(drop=True)

    return train_data, test_data


def load_embeddings(
        data_dir, models, top_n=0, vocab_filter=None, oov_vocab=None,
        padding_token='_PAD_', oov_token='_OOV_', lower=False,
        agg_method='mean'):
    """ load embedding model in memory
        if top_n > 0, load the first top_n embeddings in file
        if vocab_filter, load only tokens in vocab_filter
        if oov_vocab, add tokens in it that is not in vocab, init them using
            mean weight
        padding_token has idx 0
        oov_token has idx 1
        if lower, if a word is non-lower and has no lower version, use it as
            lower version
        if len(models) > 1, use agg_method to aggregate embeddings, either
            with 'mean' or 'concat'
    """
    logger.info("load embeddings")
    if not models:
        raise ValueError("Provide at least one model")
    vocab = {mi: {padding_token: 0, oov_token: 1} for mi in range(len(models))}
    weights = {mi: [None, None] for mi in range(len(models))}
    for mi, model in enumerate(models):
        logger.info(f"Loading embedding model {model}")
        embeddings = read_embedding_file(
            data_dir, model, top_n, vocab_filter, lower)
        for word, original_word, vector in embeddings:
            if word in vocab[mi]:
                if lower and original_word.islower():
                    # previous encounter was not lower, replace with current
                    # encouter
                    weights[mi][vocab[mi][word]] = vector
            else:
                vocab[mi][word] = len(vocab[mi])
                weights[mi].append(vector)
        if len(weights[mi]) <= 2:
            raise ValueError("No weight loaded")
        if vocab_filter:
            logger.info(
                f"Found {len(vocab[mi])}/{len(vocab_filter)} tokens in model")
        emb_size = len(weights[mi][2])
        weights[mi][vocab[mi][padding_token]] = [0.] * emb_size
        weights[mi][vocab[mi][oov_token]] = [0.] * emb_size
        mean_weight = np.mean(weights[mi], axis=0)
        weights[mi][vocab[mi][oov_token]] = mean_weight
        if oov_vocab:
            oov_to_add = oov_vocab - set(vocab[mi])
            logger.info(f"Add {len(oov_to_add)} oov tokens")
            for token in oov_to_add:
                vocab[mi][token] = len(vocab[mi])
                weights[mi].append(mean_weight)
        weights[mi] = np.array(weights[mi])
    vocab, weights = aggregate_embeddings(vocab, weights, agg_method)
    return weights, vocab


def read_embedding_file(
        data_dir: str, model: str, top_n: int = 0,
        vocab_filter: set = None, lower: bool = False
        ) -> Iterator[Tuple[str, List[float]]]:
    """ Read embedding file for different embedding models """
    errors = None
    if model == 'paragram_300_sl999/paragram_300_sl999.txt':
        errors = 'ignore'
    with open(os.path.join(data_dir, model), errors=errors) as ifs:
        for idx, line in enumerate(ifs):
            if idx == 0 and model == 'wiki-news-300d-1M/wiki-news-300d-1M.vec':
                continue  # skip header
            if idx % 10000 == 0:
                logger.debug(idx)
            if top_n and idx >= top_n:
                break
            line = line.rstrip('\n').split(' ')
            word = line[0]
            original_word = word
            if lower:
                word = word.lower()
            if vocab_filter and word not in vocab_filter:
                continue
            vector = list(map(float, line[1:]))
            yield word, original_word, vector


def aggregate_embeddings(
        vocab: Dict[int, Dict[str, int]], weights: Dict[int, np.ndarray],
        agg_method: str = 'mean'
        ) -> Tuple[Dict[str, int], Dict[int, np.ndarray]]:
    agg_vocab = {}
    for mvocab in vocab.values():
        for word in mvocab:
            if word not in agg_vocab:
                agg_vocab[word] = len(agg_vocab)
    num_models = len(vocab)
    vocab_size = len(agg_vocab)
    emb_size = len(weights[0][0])
    if agg_method == 'mean':
        agg_weights = np.empty((vocab_size, emb_size))
        for mi in range(num_models):
            mvocab, mweights = vocab[mi], weights[mi]
            for word in mvocab:
                agg_weights[agg_vocab[word]] += mweights[mvocab[word]]
        agg_weights /= num_models
    elif agg_method == 'concat':
        agg_weights = np.zeros((vocab_size, num_models*emb_size))
        for mi in range(num_models):
            mvocab, mweights = vocab[mi], weights[mi]
            for word in mvocab:
                s, e = mi*emb_size, (mi+1)*emb_size
                agg_weights[agg_vocab[word]][s:e] = mweights[mvocab[word]]
    else:
        raise ValueError(f"Unknown aggregation method {agg_method}")
    return agg_vocab, agg_weights


def get_top_terms(data):
    classes = set(data['target'])
    top_terms = {class_: Counter() for class_ in classes}
    for index, row in data.iterrows():
        tokens = row['question_text'].split()
        top_terms[row['target']].update(tokens)
    return top_terms


def preprocess_data(data, tokenizer, lower=False, lemma=False):
    """ preprocess dataset: tokenize text, optionally lowercase
    """
    logger.info(f"preprocess data (lower={lower})")
    tokenize = partial(
        _tokenize, tokenizer=tokenizer, lower=lower, lemma=lemma)
    tokenized = []
    for index, row in data.iterrows():
        if index % 100000 == 0:
            logger.debug(index)
        tokenized.append(tokenize(row['question_text']))
    data['tokenized'] = tokenized


def _tokenize(string, tokenizer, lower=False, lemma=False):
    """ tokenize, optionally lowercase """
    tokenized = tokenizer(string)
    tokens = [token.lemma_ if lemma else token.text for token in tokenized]
    if lower:
        tokens = [token.lower() for token in tokens]
    return tokens


def run_eda(train_data, test_data):
    train_data_counts = train_data.groupby('target')['qid'].count()
    analysis = {
        'train data size': len(train_data),
        'test data size': len(test_data),
        'train data classes counts': train_data_counts,
        'top terms': get_top_terms(train_data),
    }
    return analysis


def save_data(data_dir, train_data, test_data):
    """ if not saved, save training data and test_data as pickle """
    saved_files = set(['train.pkl', 'test.pkl'])
    if not saved_files < set(os.listdir(data_dir)):
        train_data.to_pickle(os.path.join(data_dir, 'train.pkl'))
        test_data.to_pickle(os.path.join(data_dir, 'test.pkl'))
        logger.info("data saved")


def map_to_input_space(
        data, vocab, max_seq_len, pad_token='_PAD_', oov_token='_OOV_'):
    """ map token to token idx
    """
    logger.info("map to input space")
    X = np.full((len(data), max_seq_len), vocab[pad_token], dtype=int)
    for index, row in data.iterrows():
        if index % 100000 == 0:
            logger.debug(index)
        for ti, token in enumerate(row['tokenized']):
            if ti >= max_seq_len:
                break
            X[index][ti] = vocab.get(token) or vocab[oov_token]
    return X


class FeedForwardNN(nn.Module):
    """ Feed-forward neural network model
    """

    def __init__(
            self, input_size, num_classes, weights, trainable_emb=False,
            hidden1=None, padding_idx=0, emb_agg='mean',
            activation='log_softmax'):
        """ weights: weights of pretrained embeddings
            if hidden1 is None, does not add hidden layer
            emb_agg: embedding aggregation method, 'sum' or 'mean'
        """
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_classes = num_classes
        self.weights = weights
        self.trainable_emb = trainable_emb
        self.padding_idx = padding_idx
        self.emb_agg = emb_agg
        self._init_embeddings()
        self.set_threshold(None, 0)
        if hidden1:
            self.input1 = nn.Linear(input_size, hidden1)
            self.hidden1 = nn.Linear(hidden1, num_classes)
        else:
            self.input1 = nn.Linear(input_size, num_classes)
            self.hidden1 = None
        self.activation = get_activation(activation)

    def _init_embeddings(self):
        num_emb, emb_size = self.weights.shape
        # by default the nn.Embedding layer outputs a double tensor
        self.embed1 = nn.Embedding(
            num_emb, emb_size, self.padding_idx,
            _weight=torch.from_numpy(self.weights)).float()
        self.embed1.weight.requires_grad = self.trainable_emb

    def forward(self, inputs, inputs_lengths=None, wide_features=None):
        """ forward pass """
        embed1 = self.embed1(inputs)
        agg_embed1 = embed1.sum(dim=1)
        if self.emb_agg == 'mean':
            if inputs_lengths is None:
                inputs_lengths = (inputs != self.padding_idx).sum(dim=1)
                inputs_lengths = inputs_lengths.to(self.device)
            inputs_lengths = inputs_lengths.to(torch.float).view(-1, 1)
            agg_embed1 /= inputs_lengths
        out = self.input1(agg_embed1)
        if self.hidden1:
            out = self.activation(out, dim=1)
            out = self.hidden1(out)
        return out

    def predict(self, inputs, lengths=None, wide_features=None):
        """ predict output class """
        out = self.predict_proba(inputs, lengths, wide_features)
        probas, predictions = torch.max(out, 1)
        if self.threshold is not None:
            default_preds = torch.ones(len(predictions)) * self.default_class
            default_preds = default_preds.long().to(self.device)
            predictions = torch.where(
                probas > self.threshold, predictions, default_preds)
        return predictions

    def predict_proba(self, inputs, lengths=None, wide_features=None):
        """ predict proba per class """
        return F.softmax(self.forward(inputs, lengths, wide_features), dim=1)

    def set_threshold(self, threshold, default_class=0):
        self.threshold = threshold
        self.default_class = default_class

    def reset_weights(self):
        """ reset model weights """
        self._init_embeddings()
        self.apply(weight_reset)


class RecurrentNN(nn.Module):
    """ Recurrent neural network model
    """

    def __init__(
            self, input_size, num_classes, weights, trainable_emb=False,
            hidden_dim_rnn=50, num_layers_rnn=1, unit_type='LSTM', dropout=0.,
            padding_idx=0, bidirectional=True, maxpooling=True,
            avgpooling=True, hidden_linear1=None, rnn_activation='relu',
            linear_activation='log_softmax', wide_dim=0):
        """ unit_type: 'LSTM' or 'GRU'
            if hidden_linear1 is not None, dim of hidden linear layer, else no
                hidden linear layer
        """
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_classes = num_classes
        self.input_size = input_size
        self.weights = weights
        self.trainable_emb = trainable_emb
        self.hidden_dim_rnn = hidden_dim_rnn
        self.num_layers_rnn = num_layers_rnn
        self.bidirectional = bidirectional
        self.maxpooling = maxpooling
        self.avgpooling = avgpooling
        self.unit_type = unit_type
        self.padding_idx = padding_idx
        self.out_rnn_dim = self.hidden_dim_rnn * (1+self.bidirectional)
        if self.maxpooling and self.avgpooling:
            # concat maxpooling and avgpooling
            self.out_rnn_dim *= 2
        self._init_embeddings()
        self.set_threshold(None, 0)
        if unit_type == 'LSTM':
            self.rnn = nn.LSTM(
                self.input_size, self.hidden_dim_rnn, self.num_layers_rnn,
                bidirectional=self.bidirectional, batch_first=True)
        elif unit_type == 'GRU':
            self.rnn = nn.GRU(
                input_size, self.hidden_dim_rnn, self.num_layers_rnn,
                bidirectional=self.bidirectional, batch_first=True)
        else:
            raise ValueError(f"Unknown unit_type {unit_type}")
        self.dropout = nn.Dropout(p=dropout)
        if hidden_linear1:
            self.linear1 = nn.Linear(self.out_rnn_dim, hidden_linear1)
            self.linear2 = nn.Linear(hidden_linear1 + wide_dim, num_classes)
        else:
            self.linear1 = nn.Linear(self.out_rnn_dim + wide_dim, num_classes)
            self.linear2 = None
        self.rnn_activation = get_activation(rnn_activation)
        self.linear_activation = get_activation(linear_activation)

    def _init_embeddings(self):
        num_emb, emb_size = self.weights.shape
        # by default the nn.Embedding layer outputs a double tensor
        self.embed1 = nn.Embedding(
            num_emb, emb_size, self.padding_idx,
            _weight=torch.from_numpy(self.weights)).float()
        self.embed1.weight.requires_grad = self.trainable_emb

    def forward(self, inputs, inputs_lengths=None, wide_features=None):
        """ forward pass """
        embed1 = self.embed1(inputs)
        if inputs_lengths is None:
            inputs_lengths = (inputs != self.padding_idx).sum(dim=1)
            inputs_lengths = inputs_lengths.to(self.device)
        # reset state at beginning of each batch
        self.hidden_rnn = self._init_hidden(len(embed1))
        # sort by length for pack_padded_sequence
        inputs_lengths, sort_idx = inputs_lengths.sort(0, descending=True)
        embed1 = embed1[sort_idx]
        # pack sequences to feed to rnn
        packed_embed1 = pack_padded_sequence(
            embed1, inputs_lengths, batch_first=True)
        rnn_out, self.hidden_rnn = self.rnn(packed_embed1)
        # unpack sequences out of rnn
        unpacked_rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        # unpacked_rnn_out: [batch_size, max(input_lengths), 2*hidden_dim_rnn]
        # put back in original order (necessary to compare to targets)
        _, unsort_idx = sort_idx.sort(0)
        if self.maxpooling or self.avgpooling:
            unpacked_rnn_out = unpacked_rnn_out[unsort_idx]
        if self.maxpooling and self.avgpooling:
            # concat maxpooling and avgpooling
            outmax, _ = torch.max(unpacked_rnn_out, dim=1)
            outavg = torch.sum(unpacked_rnn_out, dim=1)
            outavg /= inputs_lengths.float().view(-1, 1)
            out = torch.cat((outmax, outavg), dim=1)
        elif self.maxpooling:
            out, _ = torch.max(unpacked_rnn_out, dim=1)
        elif self.avgpooling:
            # use sum / input_length instead of mean because padding should
            # not contribute
            out = torch.sum(unpacked_rnn_out, dim=1)
            out /= inputs_lengths.float().view(-1, 1)
        else:
            # get the last timestep of each element
            # seems to be different than self.hidden[0]
            # idx = (in_len_batch - 1).view(-1, 1).expand(
            #   len(in_len_batch), unpacked_rnn_out.shape[2]).unsqueeze(1)
            # last_step_rnn_out = unpacked_rnn_out.gather(1, idx).squeeze(1)
            # not as good as for variable length many gradients is zero
            # last_step_rnn_out = unpacked_rnn_out[:, -1, :]
            # [batch_size, max_seq_len (of batch), (1+is_bidir)*hidden_dim_rnn]
            last_step_rnn_out = torch.cat(tuple(self.hidden_rnn[0]), dim=1)
            out = last_step_rnn_out[unsort_idx]
        # note that dropout argument of RNN layer applies dropout on all but
        # the last layer, so it is not applied if num_layers_rnn = 1
        out = self.dropout(out)
        out = self.rnn_activation(out)
        if (wide_features is not None and len(wide_features)
                and not self.linear2):
            out = torch.cat((out, wide_features), dim=1)
        out = self.linear1(out)
        if self.linear2:
            if wide_features is not None and len(wide_features):
                out = torch.cat((out, wide_features), dim=1)
            out = self.linear2(self.linear_activation(out, dim=1))
        return out

    def predict(self, inputs, lengths=None, wide_features=None):
        """ predict output class """
        out = self.predict_proba(inputs, lengths, wide_features)
        probas, predictions = torch.max(out, 1)
        if self.threshold is not None:
            default_preds = torch.ones(len(predictions)) * self.default_class
            default_preds = default_preds.long().to(self.device)
            predictions = torch.where(
                probas > self.threshold, predictions, default_preds)
        return predictions

    def predict_proba(self, inputs, lengths=None, wide_features=None):
        """ predict proba per class """
        return F.softmax(self.forward(inputs, lengths, wide_features), dim=1)

    def set_threshold(self, threshold, default_class=0):
        self.threshold = threshold
        self.default_class = default_class

    def _init_hidden(self, batch_size):
        """ init hidden state
            if unit_type == 'LSTM', returns (h0, c0) elif 'GRU' returns h0
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h0 = Variable(torch.zeros(
            self.num_layers_rnn * (1 + self.bidirectional), batch_size,
            self.hidden_dim_rnn))
        if self.unit_type == 'LSTM':
            c0 = Variable(torch.zeros(
                self.num_layers_rnn * (1 + self.bidirectional), batch_size,
                self.hidden_dim_rnn))
            return h0.to(device), c0.to(device)
        elif self.unit_type == 'GRU':
            return h0.to(device)

    def reset_weights(self):
        """ reset model weights """
        self._init_embeddings()
        self.apply(weight_reset)


class AverageEnsemble(nn.Module):

    def __init__(self, models: List[nn.Module], padding_idx=0) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.models = models

    def forward(self, inputs, *args, **kwargs) -> torch.tensor:
        outputs = [model(inputs, *args, **kwargs) for model in self.models]
        preds = torch.stack(outputs)
        return torch.mean(preds, 0)

    def predict(self, inputs, *args, **kwargs):
        """ predict output class """
        _, predictions = torch.max(self.forward(inputs, *args, **kwargs), 1)
        return predictions


def batchify(l, batch_size):
    """ return generator of batches of batch_size """
    for offset in range(0, len(l), batch_size):
        yield l[offset:offset+batch_size]


def generate_random_params(
        params_space: Dict[str, Any], num_samples: int
        ) -> Iterator[Dict[str, Any]]:
    """ yield a generator of parameters, drawing a value for each param
        :param params_space: maps param -> values
            if values is a list, draw a random element from it,
            else return values
        :param num_samples: number of combinations to generate
    """
    for _ in range(num_samples):
        params = {}
        for param, values in params_space.items():
            if type(values) == list:
                value = np.random.choice(values)
                if hasattr(value, 'item'):
                    value = value.item()
            else:
                value = values
            params[param] = value
        yield params


def _dict2sortedtuple(d):
    return tuple(sorted(d.items()))


def run_random_search(
        X, y, weights, params_space, params_score, num_samples,
        train_ratio=0.8, num_folds=None, wide_features=None):
    """ perform random search
        if num_folds is not None, train_ratio is ignored
        fill params_score dictionary with params -> scores
        if num_folds is not None, scores is a list of list of scores per fold
        per iteration, else scores is a list of scores per iteration
    """
    param_samples = generate_random_params(params_space, num_samples)
    for params in param_samples:
        tuple_params = _dict2sortedtuple(params)
        if tuple_params in params_score:
            continue
        logger.info(f"drawn params: {params}")
        model, scores = train_for_params(
            X, y, weights, params, train_ratio, num_folds, wide_features)
        params_score[tuple_params] = scores
        score = get_params_score(scores, params['num_folds'] is not None)
        logger.info(f"Max score for params: {score}")
        best_params, best_score = get_best_params(params_score)
        logger.info(f"Best params so far {best_score} using {best_params}")


def get_best_params(
        params_score: Dict[Tuple[Any], Iterable]
        ) -> Tuple[Dict[Tuple, Any], float]:
    """ return (best_params, score) given dictionary of
        params -> score per iteration
        :param params_score: params argument is a dict mapping a tuple of pairs
         (param, value) to scores
        return: params returned is a dict
    """
    best_params, best_score = {}, 0.
    for params, scores in params_score.items():
        params_dict = dict(params)
        score = get_params_score(scores, params_dict['num_folds'] is not None)
        if score > best_score:
            best_params, best_score = params_dict, score
    return best_params, best_score


def get_params_score(scores: list, cv: bool = False) -> float:
    """ return max score of all epochs
        if cv=False, scores should be a list of score per epoch, else scores
        should be a list of list of score per epoch per fold of
        cross-validation
    """
    if cv:
        folds_score = [max(fold_scores, default=0.) for fold_scores in scores]
        score = np.mean(folds_score) if folds_score else 0.
    else:
        score = max(scores, default=0.)
    return score


def train_for_params(
        X, y, weights, params, train_ratio=0.8, num_folds=None,
        wide_features=None):
    """ train classifier for given parameters
    """
    classes = sorted(set(y))
    wide_dim = wide_features.shape[1] if wide_features is not None else 0

    def model_factory():
        """ instantiate a new model
            useful e.g when using cross validation to create one model per fold
        """
        return build_model_from_params(params, len(classes), weights, wide_dim)

    def criterion_factory():
        return build_criterion_from_params(params, classes, y)

    def optimizer_factory(model):
        return build_optimizer_from_params(params, model)

    model = model_factory()
    criterion = criterion_factory()
    optimizer = optimizer_factory(model)
    if num_folds:
        scores, models = train_cv(
            X, y, model_factory, criterion_factory, optimizer_factory,
            params['num_epochs'], params['batch_size'],
            patience=params['patience'],
            min_improvement=params['min_improvement'],
            clip_grad_norm=params.get('clip_grad_norm', .0),
            tune_threshold=params['tune_threshold'],
            wide_features=wide_features)
        if params['ensemble_method'] == 'avg':
            model = AverageEnsemble(models)
        else:
            raise ValueError(
                f"Unknown ensemble method {params['ensemble_method']}")
    elif train_ratio:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio)
        train_index, test_index = next(sss.split(X, y))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        wf_train, wf_test = None, None
        if wide_features is not None:
            wf_train = wide_features[train_index]
            wf_test = wide_features[test_index]
        scores = train(
            X_train, X_test, y_train, y_test, model, criterion, optimizer,
            num_epochs=params['num_epochs'], batch_size=params['batch_size'],
            patience=params['patience'],
            min_improvement=params['min_improvement'],
            clip_grad_norm=params.get('clip_grad_norm', .0),
            tune_threshold=params['tune_threshold'], wf_train=wf_train,
            wf_test=wf_test)
    else:  # train on full dataset
        scores = train(
            X, X, y, y, model, criterion, optimizer,
            num_epochs=params['num_epochs'], batch_size=params['batch_size'],
            patience=params['patience'],
            min_improvement=params['min_improvement'],
            clip_grad_norm=params.get('clip_grad_norm', .0),
            tune_threshold=params['tune_threshold'],
            wf_train=wide_features, wf_test=wide_features)
    return model, scores


def build_model_from_params(params, num_classes, weights, wide_dim=0):
    if params['clf_model'] == 'FeedForwardNN':
        model = FeedForwardNN(
            input_size=weights.shape[1], num_classes=num_classes,
            weights=weights, trainable_emb=params['trainable_emb'],
            hidden1=params['hidden_size_1'])
    elif params['clf_model'] == 'RecurrentNN':
        model = RecurrentNN(
            input_size=weights.shape[1], num_classes=num_classes,
            weights=weights, trainable_emb=params['trainable_emb'],
            hidden_dim_rnn=params['hidden_dim_rnn'],
            unit_type=params['unit_type'], dropout=params['dropout'],
            hidden_linear1=params['hidden_linear1'], wide_dim=wide_dim)
    else:
        raise ValueError(f"Unknown model {params['clf_model']}")
    logger.info(f"Built model:\n{model}")
    return model


def build_criterion_from_params(params, classes, y):
    weight = params['loss_weight']
    if weight is not None:
        if weight == ():
            weight = compute_class_weight('balanced', classes, y)
            logger.info("Weight loss by class using class frequency")
        else:
            weight = params['loss_weight']
            logger.info(f"Weight loss by class using: {weight}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weight = torch.FloatTensor(weight).to(device)
    if params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown criterion {params['criterion']}")
    return criterion


def build_optimizer_from_params(params, model):
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=params['learning_rate'],
            momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=params['learning_rate'],
            weight_decay=params['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer {params['optimizer']}")
    return optimizer


def _tune_threshold(
        model, X_test, y_test, X_test_len, metric='f1_score', step_size=.01,
        wf_test=None):
    """ return threshold that optimize y_test """
    probas = predict_proba(
        X_test, model, lengths=X_test_len, wide_features=wf_test)
    pred_probas, predictions = torch.max(probas, dim=1)
    best_default_class, best_threshold, best_score = 0, 0, 0
    for default_class in range(model.num_classes):
        default_preds = torch.ones(len(predictions), dtype=torch.long)
        default_preds *= default_class
        for threshold in np.arange(0, 1, step_size):
            threshold_preds = torch.where(
                pred_probas > threshold, predictions, default_preds)
            score = compute_score(y_test, threshold_preds, metric=metric)
            logger.info(f"p < {threshold} -> {default_class}: {score}")
            if score > best_score:
                best_default_class = default_class
                best_threshold = threshold
                best_score = score
    return best_default_class, best_threshold, best_score


def train_cv(
        X, y, model_factory, criterion_factory, optimizer_factory, num_epochs,
        batch_size, num_folds=5, metric='f1_score', patience=10,
        min_improvement=0., clip_grad_norm=.0, tune_threshold=False,
        wide_features=None):
    """ train using cross-validation
        return (score per iteration per fold, list of trained models)
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    cv_scores, models = [], []
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        model = model_factory()
        criterion = criterion_factory()
        optimizer = optimizer_factory(model)
        logger.info(f"fold index: {fold_idx}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_wide_features, test_wide_features = None, None
        if wide_features is not None:
            train_wide_features = wide_features[train_index]
            test_wide_features = wide_features[test_index]
        fold_scores = train(
            X_train, X_test, y_train, y_test, model, criterion, optimizer,
            num_epochs, batch_size, metric, patience, min_improvement,
            clip_grad_norm, tune_threshold, train_wide_features,
            test_wide_features)
        cv_scores.append(fold_scores)
        models.append(model)
    return cv_scores, models


def train(
        X_train, X_test, y_train, y_test, model, criterion, optimizer,
        num_epochs, batch_size, metric='f1_score', patience=10,
        min_improvement=0., clip_grad_norm=0., tune_threshold=False,
        wf_train=None, wf_test=None):
    """ train model
        return score per iteration
    """
    X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    X_train_len = (X_train != model.padding_idx).sum(dim=1)
    X_test_len = (X_test != model.padding_idx).sum(dim=1)
    wf_train = wf_train if wf_train is not None else np.array([])
    wf_test = wf_test if wf_test is not None else np.array([])
    wf_train = torch.from_numpy(wf_train).float()
    wf_test = torch.from_numpy(wf_test).float()
    scores = []
    iter = 0
    best_model_sd, best_epoch = None, 0
    model.reset_weights()
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Start training...")
    for epoch in range(num_epochs):
        if len(wf_train):
            arrays = X_train, X_train_len, y_train, wf_train
        else:
            arrays = X_train, X_train_len, y_train
        arrays = unison_shuffled_copies(*arrays)
        if len(wf_train):
            X_train, X_train_len, y_train, wf_train = arrays
        else:
            X_train, X_train_len, y_train = arrays
        for offset in range(0, len(X_train), batch_size):
            inputs = X_train[offset: offset+batch_size].to(device)
            lengths = X_train_len[offset: offset+batch_size].to(device)
            targets = y_train[offset: offset+batch_size].to(device)
            wf = None
            if len(wf_train):
                wf = wf_train[offset: offset+batch_size].to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths, wf)
            loss = criterion(outputs, targets)
            loss.backward()
            if clip_grad_norm:
                # gradient clipping limits the norm of the gradient to prevent
                # the exploding gradient problems when using RNNs
                # Note that clip_grad_norm_ currently requires all parameters
                # to be of the same type
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            iter += batch_size
            if iter % 100000 == 0:
                logger.debug(
                    f"iter: {iter}, loss: {loss.item():.3f}")
        logger.debug(f"evaluate {metric} on: {Counter(y_test.tolist())}")
        score = evaluate(
            X_test, y_test, model, metric, batch_size, X_test_len, wf_test)
        if scores and score > max(scores):
            best_model_sd = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        scores.append(score)
        logger.info(f"EPOCH {epoch}: {metric} {score:.3f}")
        if early_stopping(scores, patience, min_improvement):
            logger.info(
                f"Early stopping triggered (patience {patience}, "
                f"min_improvement {min_improvement})")
            break
    if best_model_sd:
        model.load_state_dict(best_model_sd)
    model.eval()
    if tune_threshold:
        default_class, threshold, score = _tune_threshold(
            model, X_test, y_test, X_test_len, wf_test=wf_test)
        logger.info(
            f"Set tuned threshold to proba < {threshold} -> {default_class} "
            f"(score {score})")
        model.set_threshold(threshold, default_class)
        scores[best_epoch] = score
    return scores


def evaluate(
        inputs, targets, model, metric, batch_size=1000, lengths=None,
        wide_features=None):
    """ compute predictions for inputs and evaluate w.r.t to targets """
    predictions = predict(inputs, model, batch_size, lengths, wide_features)
    return compute_score(targets, predictions, metric)


def compute_score(targets, predictions, metric='f1_score'):
    if metric == 'f1_score':
        return f1_score(targets, predictions)
    else:
        raise Exception(f"unknown metric {metric}")


def _forward(
        inputs, model, model_method, batch_size=1000, lengths=None,
        wide_features=None):
    """ run model_method given inputs and model, used to process by batch if
        large number of inputs
    """
    if lengths is None:
        lengths = (inputs != model.padding_idx).sum(dim=1)
    is_training = model.training
    if is_training:
        model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions = []
    for offset in range(0, len(inputs), batch_size):
        input_batch = inputs[offset:offset+batch_size].to(device)
        length_batch = lengths[offset:offset+batch_size].to(device)
        wf_batch = None
        if wide_features is not None and len(wide_features):
            wf_batch = wide_features[offset:offset+batch_size].to(device)
        prediction_batch = model_method(
            input_batch, length_batch, wide_features=wf_batch)
        prediction_batch = prediction_batch.cpu().detach().numpy()
        predictions.extend(prediction_batch)
    if is_training:  # set back to training mode
        model.train()
    return torch.from_numpy(np.array(predictions))


def predict(inputs, model, batch_size=1000, lengths=None, wide_features=None):
    return _forward(
        inputs, model, model.predict, batch_size=batch_size, lengths=lengths,
        wide_features=wide_features)


def predict_proba(
        inputs, model, batch_size=1000, lengths=None, wide_features=None):
    return _forward(
        inputs, model, model.predict_proba, batch_size=batch_size,
        lengths=lengths, wide_features=wide_features)


def early_stopping(scores, patience, min_improvement=0.):
    """ return True if scores did not improvement within the last <patience>
        iterations, wih a minimun improvement of <min_improvement>
    """
    if len(scores) <= patience:
        return False
    best_score_outside_patience = max(scores[:-patience])
    best_score_in_patience = max(scores[-patience:])
    improvement = best_score_in_patience - best_score_outside_patience
    return improvement < min_improvement


def weight_reset(m):
    """ reset weights of model, used between runs
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def get_activation(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'log_softmax':
        return F.log_softmax
    else:
        raise ValueError(f"Unknown activation {activation}")


def downsample(data, downsample_ratio=None, max_imbalance_ratio=None):
    """ downsample data, either keeping the same proportion between classes
        using <downsample_ratio> or by enforcing a maximum ratio of imbalance
        with <max_imbalance_ratio>.
        max_imbalance_ratio=2. means that the most common class cannot have
        more than twice the number of datapoints than the least common class
    """
    if max_imbalance_ratio:
        counts_per_class = dict(data.groupby('target')['qid'].count())
        downsampled_dfs = {}
        min_class_count = min(counts_per_class.values(), default=0.)
        min_class_count = int(min_class_count * max_imbalance_ratio)
        logger.info(f"Downsampling current counts {counts_per_class}, "
                    f"keeping a max ratio of {max_imbalance_ratio} between "
                    f"classes")
        for class_, count in counts_per_class.items():
            downsampled_dfs[class_] = data.loc[data['target'] == class_] \
                .sample(min(count, min_class_count))
        data = pd.concat(downsampled_dfs.values()).reset_index(drop=True)
    if downsample_ratio and downsample_ratio < 1.:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=downsample_ratio)
        data_index, _ = next(sss.split(data, data['target']))
        data = data.iloc[data_index]
    # shuffle
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def unison_shuffled_copies(*arrays):
    """ shuffle numpy arrays keeping the alignment between them """
    if not arrays:
        return arrays
    assert all(len(a) == len(arrays[0]) for a in arrays)
    p = np.random.permutation(len(arrays[0]))
    return (a[p] for a in arrays)


def build_vocab(data, vocab_size=0):
    """ return set containing all distinct tokens appearing in the data
    """
    vocab = Counter()
    for tokenized in data['tokenized'].values:
        vocab.update(tokenized)
    if vocab_size > 0:
        vocab = set(list(zip(*vocab.most_common(vocab_size)))[0])
    else:
        vocab = set(vocab)
    return vocab


def correct_text(data: pd.DataFrame, corrections: Dict[str, str]) -> None:
    corrected_text = []
    for index, row in data.iterrows():
        if index % 100000 == 0:
            logger.debug(index)
        corrected_text.append(_correct_text(row['question_text'], corrections))
    data['question_text'] = corrected_text


def _correct_text(text: str, corrections: Dict[str, str]) -> str:
    for error, correction in corrections.items():
        text = text.replace(error, correction)
    return text


def get_corrections() -> Dict[str, str]:
    mispell_dict = {
        'colour': 'color', 'centre': 'center', 'favourite': 'favorite',
        'travelling': 'traveling', 'counselling': 'counseling',
        'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
        'organisation': 'organization', 'wwii': 'world war 2',
        'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
        'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist',
        'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
        'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',
        'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
        'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
        "mastrubating": 'masturbating', 'pennis': 'penis',
        'Etherium': 'Ethereum', 'narcissit': 'narcissist',
        'bigdata': 'big data', '2k17': '2017', '2k18': '2018',
        'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
        'airhostess': 'air hostess', "whst": 'what',
        'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
        'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
    corrections = {**mispell_dict}
    return corrections


def compute_wide_features(
        data: pd.DataFrame, vocab: set, max_seq_len: int) -> np.ndarray:
    wide_features_f = [
        lambda r, l, t, vocab=vocab: len(set(t) - vocab),  # number of oov
        lambda r, l, t: int(r[0].isupper()),  # start with uppercase
        lambda r, l, t: sum(w.isupper() for w in r.split(' ')),  # number upper
        lambda r, l, t: int('!!' in r),  # start with uppercase
        lambda r, l, t: int(r[0].isupper()),  # start with uppercase
    ]
    sensitive_topics = [
        'india', 'america', 'europe', 'africa', 'asia', 'south america',
        'muslim', 'atheist', 'islam', 'christian', 'religio', 'hindu', 'allah',
        'jesus',
        'black', 'arab', 'aryan', 'race',
        'vegan', 'vegetarian',
        'homosexual', 'gay', 'lesbian', 'feminist', 'transgender',
        'people', 'men', 'women', 'hate', 'why', 'girls', 'boys', 'child',
        'trump', 'democrats', 'liberal', 'obama',
        'job', 'good', 'think', 'feel', 'like', 'lol', 'money',
        'penis', 'pussy', 'sex', 'dick', 'boobs', 'tits', 'rape'
    ]
    insults = [
        'arse', 'ass', 'cock', 'bitch', 'fucker', 'fuck', 'suck', 'sucker',
        'dickhead', 'dick-head', 'goon', 'motherfucker', 'shit', 'slut',
        'weirdo', 'loser', 'junkie',
    ]
    countries = [
        'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola',
        'Argentina', 'Armenia', 'Australia', 'Austria',
        'Azerbaijan', 'Bahamas', 'Bangladesh',
        'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Bolivia',
        'Bosnia Herzegovina', 'Botswana', 'Brazil', 'Bulgaria',
        'Burkina', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde',
        'Chad', 'Chile', 'China', 'Colombia', 'Comoros',
        'Congo', 'Congo', 'Costa Rica', 'Croatia', 'Cuba',
        'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica',
        'Dominican Republic', 'East Timor', 'Ecuador', 'Egypt', 'El Salvador',
        'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji',
        'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana',
        'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guyana',
        'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia',
        'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast',
        'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',
        'North Korea', 'South Korea', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos',
        'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein',
        'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi',
        'Malaysia', 'Maldives', 'Mali', 'Malta',
        'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova',
        'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
        'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands',
        'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman',
        'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay',
        'Peru', 'Philippines', 'Filipin', 'Poland', 'Portugal', 'Qatar',
        'Romania', 'Russia', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Serbia',
        'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
        'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 'Spain',
        'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland',
        'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo',
        'Tunisia', 'Turkey', 'Turkmenistan',
        'Uganda', 'Ukraine', 'United Arab Emirates',
        'United Kingdom', 'England', 'United States', 'Uruguay', 'Uzbekistan',
        'Vanuatu', 'Vatican', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia',
        'Zimbabwe'
    ]
    countries = [c.lower() for c in countries]
    topics = sensitive_topics + insults + countries
    for topic in topics:
        wide_features_f.append(lambda r, l, t, topic=topic: int(topic in l))
    bucket_size = int(max_seq_len/10)
    for offset in range(0, max_seq_len, bucket_size):
        s, e = offset, offset + bucket_size
        wide_features_f.append(
            lambda r, l, t, s=s, e=e: int(s <= len(t) < e))
    wide_features = np.empty((len(data), len(wide_features_f)))
    for index, row in data.iterrows():
        raw, lowered = row['original_text'], row['original_text'].lower()
        tokens = row['tokenized']
        row_wide_features = np.array(
            [f(raw, lowered, tokens) for f in wide_features_f])
        wide_features[index] = row_wide_features
    return wide_features


def get_saved_best_params():
    best_params = {
        'batch_size': 1024,
        'clf_model': 'RecurrentNN',
        'clip_grad_norm': 0.25,
        'criterion': 'CrossEntropyLoss',
        'downsample': 1.0,
        'dropout': 0.4,
        'emb_agg_method': 'concat',
        'embedding_models': (
            'glove.840B.300d/glove.840B.300d.txt',
            'paragram_300_sl999/paragram_300_sl999.txt',
            'wiki-news-300d-1M/wiki-news-300d-1M.vec'),
        'embeddings_top_n': 0,
        'hidden_dim_rnn': 150,
        'hidden_linear1': 150,
        'learning_rate': 0.001,
        'loss_weight': None,
        'correct_text': True,
        'lower': False,
        'lemma': False,
        'max_imbalance_ratio': 0.0,
        'max_seq_len': 50,
        'min_improvement': 0.01,
        'momentum': 0.2,
        'num_epochs': 5,
        'num_folds': None,
        'optimizer': 'Adam',
        'patience': 10,
        'seed': 828600365,
        'spacy_model': 'en_core_web_sm',
        'train_ratio': 0.,
        'trainable_emb': True,
        'unit_type': 'LSTM',
        'vocab_size': 100000,
        'weight_decay': 1e-06,
        'tune_threshold': True,
        'wide_features': True,
    }
    best_params['_num_epochs'] = 15
    best_params['_thresh_class_1'] = .6
    return best_params


def get_params_space():
    PARAMS_SPACE = {
        'seed': np.random.randint(1E9),
        'correct_text': True,
        'lower': False,
        'lemma': False,
        'downsample': 1.,  # None, 0 or 1 to ignore
        'max_imbalance_ratio': 0.,
        'max_seq_len': 50,
        'embedding_models': (
            'glove.840B.300d/glove.840B.300d.txt',
            'paragram_300_sl999/paragram_300_sl999.txt',
            'wiki-news-300d-1M/wiki-news-300d-1M.vec'
        ),
        'vocab_size': 100000,
        'embeddings_top_n': 0,
        'emb_agg_method': 'concat',
        'spacy_model': 'en_core_web_sm',
        'batch_size': [2**i for i in range(8, 12)],
        'weight_decay': [10**i for i in range(-6, -4)],
        'momentum': np.arange(0., 0.91, 0.1).tolist(),
        'num_epochs': 5,
        'patience': 10,
        'min_improvement': 1E-2,
        'criterion': "CrossEntropyLoss",
        'loss_weight': [None, (), (1., 2.)],  # () means balanced with classes
        'optimizer': ["SGD", "Adam"],
        'trainable_emb': True,
        'train_ratio': 0.,
        'num_folds': None,
        'tune_threshold': True,
        'wide_features': True,
        'clf_model': 'RecurrentNN',
    }
    if PARAMS_SPACE['num_folds'] is not None:
        PARAMS_SPACE['ensemble_method'] = 'avg'
    if PARAMS_SPACE['clf_model'] == 'FeedForwardNN':
        PARAMS_SPACE.update({
            'hidden_size_1': [0, 32, 64, 128],
            'emb_agg': ['mean'],
            'learning_rate': [i*1E-4 for i in [1, 2, 5, 10]],
        })
    elif PARAMS_SPACE['clf_model'] == 'RecurrentNN':
        PARAMS_SPACE.update({
            'unit_type': ['LSTM', 'GRU'],
            'hidden_dim_rnn': [32, 64, 128],
            'hidden_linear1': [32, 64, 128],
            'learning_rate': [i*1E-2 for i in [.5, 1, 2]],
            'dropout': np.arange(0., 0.51, 0.1).tolist(),
            'clip_grad_norm': [0., 0.25]
        })
    else:
        raise ValueError(
            f"Unknown classifier model {PARAMS_SPACE['clf_model']}")
    return PARAMS_SPACE


def main(num_samples=0):
    SUBMIT = not num_samples
    PARAMS_SPACE = get_params_space()
    best_params = get_saved_best_params()
    if SUBMIT:
        PARAMS_SPACE = best_params
    set_seeds(PARAMS_SPACE['seed'])
    logger.info(f"SEED: {PARAMS_SPACE['seed']}")
    logger.info(f"torch.initial_seed(): {torch.initial_seed()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Use device {device}")
    data_dir = get_data_dir()
    embed_dir = os.path.join(data_dir, 'embeddings/')
    log_dir = get_log_dir()
    train_data, test_data = load_data(data_dir)
    train_data = downsample(
        train_data, PARAMS_SPACE['downsample'],
        PARAMS_SPACE['max_imbalance_ratio'])
    train_data['original_text'] = train_data['question_text']
    test_data['original_text'] = test_data['question_text']
    if PARAMS_SPACE['correct_text']:
        corrections = get_corrections()
        correct_text(train_data, corrections)
    nlp = spacy.load(
        PARAMS_SPACE['spacy_model'], disable=['tagger', 'parser', 'ner'])
    preprocess_data(
        train_data, nlp, PARAMS_SPACE['lower'], PARAMS_SPACE['lemma'])
    preprocess_data(
        test_data, nlp, PARAMS_SPACE['lower'], PARAMS_SPACE['lemma'])
    train_vocab = build_vocab(train_data, PARAMS_SPACE['vocab_size'])
    weights, vocab = load_embeddings(
        embed_dir, models=PARAMS_SPACE['embedding_models'],
        vocab_filter=train_vocab, oov_vocab=train_vocab,
        lower=PARAMS_SPACE['lower'], top_n=PARAMS_SPACE['embeddings_top_n'],
        agg_method=PARAMS_SPACE['emb_agg_method'])
    train_wide_features = None
    if PARAMS_SPACE['wide_features']:
        train_wide_features = compute_wide_features(
            train_data, set(vocab), PARAMS_SPACE['max_seq_len'])
    logger.info(f"Vocab size: {len(vocab)}")
    emb_size = weights.shape[1]
    num_classes = len(set(train_data['target']))
    X_train = map_to_input_space(
        train_data, vocab, PARAMS_SPACE['max_seq_len'])
    y_train = train_data['target'].values
    if not SUBMIT:
        params_score = {}
        run_random_search(
            X_train, y_train, weights, PARAMS_SPACE, params_score,
            num_samples, PARAMS_SPACE['train_ratio'],
            PARAMS_SPACE['num_folds'], wide_features=train_wide_features)
        best_params, score = get_best_params(params_score)
        model, scores = train_for_params(
            X_train, y_train, weights, best_params, best_params['train_ratio'],
            best_params['num_folds'], wide_features=train_wide_features)
        # analysis = run_eda(train_data, test_data)
    else:
        logger.info("final training")
        model, scores = train_for_params(
            X_train, y_train, weights, best_params, best_params['train_ratio'],
            best_params['num_folds'], wide_features=train_wide_features)
        logger.info("preprocess and predict target on test set")
        if PARAMS_SPACE['correct_text']:
            correct_text(test_data, corrections)
        test_wide_features = None
        if PARAMS_SPACE['wide_features']:
            test_wide_features = compute_wide_features(
                test_data, set(vocab), PARAMS_SPACE['max_seq_len'])
        X_test = map_to_input_space(
            test_data, vocab, PARAMS_SPACE['max_seq_len'])
        X_test = torch.from_numpy(X_test)
        test_wide_features = torch.from_numpy(test_wide_features).float()
        test_data['prediction'] = predict(
            X_test, model, wide_features=test_wide_features)
        test_data[['qid', 'prediction']].to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main(0)

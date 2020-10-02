'''
This file is uploaded to Kaggle, see https://www.kaggle.com/soulmachine/jigsaw-utils-kaggle.
'''
from __future__ import absolute_import, division, print_function

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union

import dask.bag as db
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer


logger = logging.getLogger(__name__)

LOCAL_ROOT_DIR = '../input'

NAN_WORD = "_NAN_"

Y_COLUMNS = ['target']
Y_AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
EXTRA_PARAMS_FOR_BERT = ['max_seq_length', 'y_aux_columns', 'num_labels', 'do_lower_case']


#################### preprocessing.py ####################
import re

from typing import List, Tuple


replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'(I|i)\'m', 'I am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\\g<1> will'),
    (r'(\w+)n\'t', '\\g<1> not'),
    (r'(\w+)\'ve', '\\g<1> have'),
    (r'(\w+)\'s been', '\\g<1> has been'),
    (r'(\w+)\'s', '\\g<1> is'),
    (r'(\w+)\'re', '\\g<1> are'),
    (r'(\w+)\'d', '\\g<1> would')
]

class RegexpReplacer(object):
    '''Expand contraction, e.g., can't to cannot, I'm to I am, etc.
    '''
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl)
                         in
                         patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

REGEX_REPLACER = RegexpReplacer()

def preprocess(comment: str, preprocess_id: str)->str:
    if preprocess_id == 'none' or preprocess_id == 'none_head_tail':
        return comment
    elif preprocess_id == 'basic_clean' or preprocess_id == 'basic_clean_head_tail':
        return re.sub(' +', ' ', comment.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip())
    elif preprocess_id == 'expand' or preprocess_id == 'expand_head_tail':
        cleaned = re.sub(' +', ' ', comment.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip())
        return REGEX_REPLACER.replace(cleaned)

def pad_comment_ids(comment_ids: str, max_seq_length: int)->List[int]:
    arr = comment_ids.split(' ')
    ints = list(map(int, arr))
    return ints + [0] * (max_seq_length - len(ints))

def to_preprocess_full_id(preprocess_id: str, max_seq_length: int, vocab_md5: str)->str:
    return f'{preprocess_id}--seq_{max_seq_length}--vocab_{vocab_md5}'

def from_preprocess_full_id(preprocess_full_id: str)->Tuple[str, int, str]:
    ss = preprocess_full_id.split('--')
    assert len(ss) == 3
    preprocess_id = ss[0]
    max_seq_length = int(ss[1].split('_')[1])
    vocab_md5 = ss[2].split('_')[1]
    return (preprocess_id, max_seq_length, vocab_md5)


#################### common_utils.py ####################
import hashlib
import sys
import tqdm as _tqdm


def is_notebook():
    return  'ipykernel' in sys.modules

tqdm = _tqdm.tqdm_notebook if is_notebook() else _tqdm.tqdm

def md5(fname: str)->str:
    '''Calculate the md5 of a file, the same as the Linux command md5sum.

    Copied from https://stackoverflow.com/a/3431838/381712
    '''
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

#################### inference_utils.py ####################
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.utils.data

import argparse
import gc
import logging
import os
import platform
import shutil
import subprocess
import time


logger = logging.getLogger(__name__)

PARAMS_NEEDED_BY_BOTH_TRAINING_AND_INFERENCE = [
    'model_id',
    'preprocess_id',
    'preprocess_full_id',
    'model_root_dir',
    'model_full_id',
    'num_gpus',
    'total_batch_size',
    'seed',
    'debug',
    'epochs_dir_local',
]

PARAMS_NEEDED_BY_INFERENCE = [
    'test_dataset_id',
    'eval_batch_size',
    'epoch',
    'epoch_dir_local',
    'prediction_output_file_local',
]


def to_eval_model(model):
    model.eval()
    for param in model.parameters():  # Saves GPU memory a lot !!!
        param.requires_grad = False
    return model

def enrich_args_common(args: argparse.Namespace)->argparse.Namespace:
    for key in ['model_root_dir', 'model_id', 'model_full_id', 'debug']:
        if key not in args:
            raise ValueError(key)

    # see dataset https://www.kaggle.com/soulmachine/jigsaw-pytorch-bert-model-dir
    if args.debug and not args.model_root_dir.endswith('-debug'):
        args.model_root_dir = args.model_root_dir + '-debug'

    args.epochs_dir_local = f'{os.environ["LOCAL_ROOT_DIR"]}/{args.model_root_dir}/{args.model_full_id}'
    if os.environ.get("GCS_ROOT_DIR"):
        args.epochs_dir_gcs = f'{os.environ["GCS_ROOT_DIR"]}/{args.model_root_dir}/{args.model_full_id}'

    args.hostname = platform.node()
    args.num_gpus = torch.cuda.device_count()
    if 'train_dataset_id' in args:
        args.total_batch_size = args.train_batch_size * args.num_gpus
    else:
        args.total_batch_size = args.eval_batch_size * args.num_gpus

    if 'epoch' in args:
        step_str = f'-step-{args.step}' if 'step' in args and args.step else ''
        epoch_dir_name = f'epoch-{args.epoch}' + step_str
        args.epoch_dir_local = f'{args.epochs_dir_local}/{epoch_dir_name}'
        if os.environ.get("GCS_ROOT_DIR"):
            args.epoch_dir_gcs = f'{args.epochs_dir_gcs}/{epoch_dir_name}'

        if 'test_dataset_id' in args:
            args.prediction_output_file_local = f'{os.environ["LOCAL_ROOT_DIR"]}/{args.model_root_dir}-predictions-{args.test_dataset_id}/{args.model_full_id}/{epoch_dir_name}.csv'
            if os.environ.get("GCS_ROOT_DIR"):
                args.prediction_output_file_gcs = f'{os.environ["GCS_ROOT_DIR"]}/{args.model_root_dir}-predictions-{args.test_dataset_id}/{args.model_full_id}/{epoch_dir_name}.csv'

def predict(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.TensorDataset,
    predict_func: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    args: argparse.Namespace,
)->np.ndarray:
    assert model.training == False
    device = torch.device('cuda')
    batch_size=args.total_batch_size
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    y_preds = np.zeros((len(test_dataset), args.num_labels))

    for step, (x_batch, ) in tqdm(enumerate(test_loader), total=len(test_loader)):
        y_pred = predict_func(model, x_batch.to(device))
        #y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        assert y_pred.shape[1] == args.num_labels
        y_preds[step*batch_size:(step+1)*batch_size]=y_pred.detach().cpu().numpy()
    del test_loader
    return torch.sigmoid(torch.tensor(y_preds)).numpy()

def predict_epoch_common(
    epoch: int,
    args: argparse.Namespace,
    load_cur_model_func: Callable[[int, argparse.Namespace], torch.nn.Module],
    test_df: pd.DataFrame,
    calc_X_func: Callable[[pd.DataFrame, argparse.Namespace], np.ndarray],
    predict_func: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    save_preds_func: Callable[[pd.DataFrame, np.ndarray], bool],
    outer_tq=None,
)->bool:
    assert epoch == args.epoch
    for key in PARAMS_NEEDED_BY_BOTH_TRAINING_AND_INFERENCE + PARAMS_NEEDED_BY_INFERENCE:
        assert key in args

    try:
        output_dir_local = os.path.dirname(args.prediction_output_file_local)
        if 'epoch_dir_gcs' in args:
            from gcs_utils import gcs_path_exists
            if gcs_path_exists(args.prediction_output_file_gcs):
                return False
            else:
                if os.path.exists(args.prediction_output_file_local):
                    logger.warning(f'{args.prediction_output_file_local} already exists, deleting it')
                    shutil.rmtree(output_dir_local)
        os.makedirs(output_dir_local, exist_ok=True)

        begin_time = time.time()
        logger.debug(f'Predicting on model {args.epoch_dir_local}')

        X_test = calc_X_func(test_df, args)
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test))

        model = load_cur_model_func(epoch, args)
        model = to_eval_model(model)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)  # model.training becomes true again
            model = to_eval_model(model)

        y_test = predict(model, test_dataset, predict_func, args)

        save_preds_func(test_df, y_test, args)
        if os.environ.get('GCS_ROOT_DIR'): # Upload to GCS
            output_dir_gcs = os.path.dirname(args.prediction_output_file_gcs)
            logger.debug(subprocess.check_output(
                f'gsutil -m cp {output_dir_local}/* {output_dir_gcs}',
                shell=True).decode('utf-8'))
        end_time = time.time()
        logger.debug(f'Time elapsed {int(end_time-begin_time)}s')
        return True
    except RuntimeError as ex:
        error_msg = str(ex)
        logger.error(error_msg)
        return False
    finally:
        del y_test
        del model
        del test_dataset
        del X_test
        gc.collect()
        torch.cuda.empty_cache()

    return False

#################### Common ####################
def seed_everything(seed=73):
    '''
      Make PyTorch deterministic.
    '''    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

def args_to_model_full_id(args: Dict[str, Any])->str:
    for key, value in args.items():
        if type(value) == str and '.' in value and '/' not in value:
            raise ValueError(f'{key}:{value}')

    with open('config/vocab_to_md5.json', 'rt') as infile:
        vocab_to_md5 = json.load(infile)
    vocab_md5 = vocab_to_md5[f'{LOCAL_ROOT_DIR}/bert-pretrained-pytorch/{args["model_id"]}/vocab.txt']
    preprocess_full_id = preprocessing.to_preprocess_full_id(args['preprocess_id'], args['max_seq_length'], vocab_md5)

    y_aux_hash = hashlib.md5(','.join(args['y_aux_columns']).encode('utf-8')).hexdigest()[0:7]
    model_full_id = f'{args["train_dataset_id"]}--{preprocess_full_id}--{args["model_id"]}--{args["loss_func_id"]}' + \
           f'--y_aux_hash-{y_aux_hash}--{"fp16" if args["fp16"] else "fp32"}' + \
           f'--batch-{args["train_batch_size"]}--acc_steps-{args["gradient_accumulation_steps"]}--lr_{args["lr"]}--seed-{args["seed"]}'
    return (model_full_id, preprocess_full_id)

def model_full_id_to_args(model_full_id: str)->Dict[str, Any]:
    assert '/' not in model_full_id
    args = {}
    ss = model_full_id.split('--')
    assert len(ss) == 12
    args['train_dataset_id'] = ss[0]
    args['preprocess_id'] = ss[1]
    args['max_seq_length'] = int(ss[2].split('_')[1])
    args['vocab_md5'] = ss[3].split('_')[1]
    args['model_id'] = ss[4]
    args['loss_func_id'] = ss[5]
    args['y_aux_hash'] = ss[6].split('-')[1]
    args['fp16'] = (ss[7]) == 'fp16'
    args['train_batch_size'] = int(ss[8].split('-')[1])
    args['gradient_accumulation_steps'] = int(ss[9].split('-')[1])
    args['lr'] = float(ss[10].split('_')[1])
    args['seed'] = int(ss[11].split('-')[1])

    preprocess_full_id = f'{args["preprocess_id"]}--seq_{args["max_seq_length"]}--vocab_{args["vocab_md5"]}'
    args['preprocess_full_id'] = preprocess_full_id
    return args


#################### Data ####################

def convert_lines(lines, max_seq_length, tokenizer, head_tail=False):
    '''
      Converting the lines to BERT format.
      
      Copied from https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
    '''
    max_seq_length -= 2  # CLS, SEP
    all_tokens = []
    longer = 0
    for text in tqdm(lines):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            if head_tail:
                tokens_a = tokens_a[:max_seq_length//2] + tokens_a[-1-(max_seq_length - max_seq_length//2):-1]
            else:
                tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    logger.debug(f'longer: {longer}')
    return np.array(all_tokens)


def convert_lines_parallel(i, lines, max_seq_length, tokenizer, head_tail=False):
    total_lines = len(lines)
    num_lines_per_thread = total_lines // os.cpu_count() + 1
    lines = lines[i * num_lines_per_thread : (i+1) * num_lines_per_thread]
    return convert_lines(lines, max_seq_length, tokenizer, head_tail)

def clean_data(df, args):
    '''
       Cleaning, agnostic to model params.
    '''
    # see https://stackoverflow.com/a/43004612/381712
    df.update(df.select_dtypes(include=[np.number]).fillna(0))
    # Make sure all comment_text values are strings
    df['comment_text'] = df['comment_text'].astype(str).fillna(NAN_WORD)
    if args.preprocess_id != 'none':
        df['comment_text'] = df['comment_text'].apply(lambda s: preprocess(s, args.preprocess_id))

    return df

def calc_X(df: pd.DataFrame, tokenizer, args: argparse.Namespace):
    '''Calculate X only.
     This function is idempotent, re-enterable, and stateless.
    '''
    logger.info(f'Calculating X_test')
    X = df["comment_text"]
    X = np.vstack(db.from_sequence(list(range(os.cpu_count()))).map(
        lambda i: convert_lines_parallel(i, X, args.max_seq_length, tokenizer, args.preprocess_id.endswith('head_tail'))
    ).compute())
    return X


#################### Inference ####################
def create_inference_args(str_list):
    '''
      A lite version of args at https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L565
    '''
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_root_dir",
                        default='jigsaw-model-dir',
                        type=str,
                        required=True,
                        help="The parent dir of model dirs.")
    parser.add_argument("--model_full_id",
                        default=None,
                        type=str,
                        required=True,
                        help="The model full ID.")
    parser.add_argument("--epoch",
                        default=0,
                        type=int,
                        required=True,
                        help="The epoch number.")

    ## Other parameters
    parser.add_argument("--step",
                        type=int,
                        help="The step number.")
    parser.add_argument("--test_dataset_id",
                        default='jigsaw_2019_test',
                        type=str,
                        help="The unique ID of the dataset.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help="Debug mode, use very little data")

    args = parser.parse_args(str_list)

    return args

def enrich_inference_args(args):
    assert 'model_root_dir' in args and 'model_full_id' in args
    assert 'epoch' in args
    args_from_model_full_id = model_full_id_to_args(os.path.basename(args.model_full_id))
    args.model_id = args_from_model_full_id['model_id']

    enrich_args_common(args)

    with open(os.path.join(args.epoch_dir_local, 'training_params.json'), 'rt') as infile:
        training_params = json.load(infile)

    keys_needed_by_inference = PARAMS_NEEDED_BY_BOTH_TRAINING_AND_INFERENCE + \
        PARAMS_NEEDED_BY_INFERENCE + EXTRA_PARAMS_FOR_BERT
    for key in keys_needed_by_inference:
        if key in args:
            continue
        if key in args_from_model_full_id and key in training_params and args_from_model_full_id[key] != training_params[key]:
            raise ValueError(f'{key}: {training_params[key]}')
        if key in training_params:
            args.__dict__[key] = training_params[key]
    assert args.do_lower_case == ('uncased' in args.model_id)

def check_inference_args(args):
    all_params_needed_by_infer = PARAMS_NEEDED_BY_BOTH_TRAINING_AND_INFERENCE + \
        PARAMS_NEEDED_BY_INFERENCE + EXTRA_PARAMS_FOR_BERT
    for param in all_params_needed_by_infer:
        if param not in args:
            raise ValueError(param)

    assert ('uncased' in args.model_id) == args.do_lower_case
    assert os.path.exists(args.epoch_dir_local)
    if 'GCS_ROOT_DIR' in os.environ:
        from gcs_utils import gcs_path_exists
        assert gcs_path_exists(args.epoch_dir_gcs)

def load_model(model_dir, num_labels):
    assert os.path.exists(model_dir)
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        cache_dir=None,
    )

    model.to(torch.device('cuda'))
    return model

def load_prev_epoch_model(epoch: int, args: argparse.Namespace)->torch.nn.Module:
    assert epoch == args.epoch
    # get model_dir_local
    if args.epoch == 0:  # Load BERT model
        model_dir_local = os.path.join(os.environ['LOCAL_ROOT_DIR'], 'bert-pretrained-pytorch', args.model_id)
    else:  # Load previous model
        model_dir_local = os.path.join(args.epochs_dir_local, f'epoch-{epoch-1}')

    copy_from_gcs = False
    if not os.path.exists(model_dir_local):
        if 'GCS_ROOT_DIR' in os.environ:
            from gcs_utils import rsync, gcs_path_exists
            # get model_dir_gcs
            if args.epoch == 0:  # Load BERT model
                model_dir_gcs = os.path.join('gs://kagg1e', 'bert-pretrained-pytorch', args.model_id)
            else:  # Load previous model
                model_dir_gcs = os.path.join(args.epochs_dir_gcs, f'epoch-{epoch-1}')

            if gcs_path_exists(model_dir_gcs):
                rsync(model_dir_gcs, model_dir_local)
                copy_from_gcs = True
            else:
                raise ValueError(f'{model_dir_gcs} does NOT exist')
        else:
            raise ValueError(f'"GCS_ROOT_DIR" not in os.environ ')
    assert os.path.exists(model_dir_local)
    model = load_model(model_dir_local, args.num_labels)
    if copy_from_gcs:
        shutil.rmtree(model_dir_local)
    return model

def load_cur_epoch_model(epoch: int, args: argparse.Namespace)->torch.nn.Module:
    assert epoch == args.epoch
    epoch_name = f'epoch-{args.epoch}-step-{args.step}' if 'step' in args and args.step else f'epoch-{args.epoch}'
    model_dir_local = os.path.join(args.epochs_dir_local, epoch_name)
    copy_from_gcs = False
    if not os.path.exists(model_dir_local):
        if 'GCS_ROOT_DIR' in os.environ:
            from gcs_utils import rsync, gcs_path_exists
            model_dir_gcs = os.path.join(args.epochs_dir_gcs, epoch_name)
            if gcs_path_exists(model_dir_gcs):
                rsync(model_dir_gcs, model_dir_local)
                copy_from_gcs = True
            else:
                raise ValueError(f'{model_dir_gcs} does NOT exist')
        else:
            raise ValueError(f'"GCS_ROOT_DIR" not in os.environ ')
    assert os.path.exists(model_dir_local)
    model = load_model(model_dir_local, args.num_labels)
    if copy_from_gcs:
        shutil.rmtree(model_dir_local)
    return model

def load_eval_model(model_dir, num_labels):
    model = load_model(model_dir, num_labels)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(torch.device('cuda'))
    return to_eval_model(model)

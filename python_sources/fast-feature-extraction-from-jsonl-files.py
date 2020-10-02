"""
This kernel performs converting jsonl files to csv on the fly without loading all data in memory.
It also uses multiprocessing approach to speedup processing on CPUs with multiple cores.
Locally (8 cores, SSD) it takes 35 seconds to process both train_matches.jsonl and test_matches.jsonl.
With default features the difference is not significant,
but when you add more features it will save you a lot of time.
"""

import csv
import time
from collections import OrderedDict
from concurrent import futures
from contextlib import contextmanager
from itertools import chain
from itertools import islice
from pathlib import Path

import ujson as json
from tqdm import tqdm

PATH_TO_DATA = Path('../input')

MATCH_FEATURES = [
    ('game_mode', lambda m: m['game_mode']),
]

MATCH_FIELDS = [
    'match_id_hash', 'players', 'targets',
    'game_time', 'game_mode', 'lobby_type', 'objectives', 'chat'
]


PLAYER_FIELDS = [
    'hero_id',

    'kills',
    'deaths',
    'assists',
    'denies',

    'gold',
    'lh',
    'xp',
    'health',
    'max_health',
    'max_mana',
    'level',

    'x',
    'y',

    'stuns',
    'creeps_stacked',
    'camps_stacked',
    'rune_pickups',
    'firstblood_claimed',
    'roshans_killed',
    'obs_placed',
    'sen_placed',
]


def extract_features(match):
    match_id_hash = match['match_id_hash']
    row = [
        ('match_id_hash', match_id_hash),
        ('target', extract_targets(match)),
    ]
    append = row.append
    for field, f in MATCH_FEATURES:
        append((field, f(match)))
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = f'r{slot + 1}'
        else:
            player_name = f'd{slot - 4}'
        for field in PLAYER_FIELDS:
            append((f'{player_name}_{field}', player[field]))

        radiant_tower_kills = 0
        dire_tower_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 2:
                    radiant_tower_kills += 1
                if objective['team'] == 3:
                    dire_tower_kills += 1
        append(('r_tower_kills', radiant_tower_kills))
        append(('d_tower_kills', dire_tower_kills))
        append(('diff_tower_kills', radiant_tower_kills - dire_tower_kills))
    return OrderedDict(row)


def extract_targets(match):
    if 'targets' in match:
        return 1 if match['targets']['radiant_win'] else 0
    else:
        return ''


def extract_features_from_line(line):
    return extract_features(json.loads(line))


def read_lines(reader, offset, limit):
    for i, line in reader:
        if offset and i < offset:
            continue
        if limit and i >= offset + limit:
            return
        yield line


def get_chunks(iterable, size):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def process(fin, fout, offset=0, limit=0, chunk_size=64):
    reader = iter(enumerate(fin))
    _, line = next(reader)
    features = extract_features(json.loads(line))
    fieldnames = tuple(features.keys())
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    if offset == 0:
        writer.writerow(features)
    chunks = get_chunks(tqdm(read_lines(reader, offset, limit), total=offset + limit or None), chunk_size)
    with futures.ProcessPoolExecutor() as pool:
        for features in chain.from_iterable(map(lambda x: pool.map(extract_features_from_line, x), chunks)):
            writer.writerow(features)


@contextmanager
def timer(name):
    print(f'[{name}]')
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.1f} s')


with timer('Extract features'):
    with (PATH_TO_DATA / 'train_matches.jsonl').open() as fin, open('train_matches.csv', 'w') as fout:
        process(fin, fout, limit=39675)
    with (PATH_TO_DATA / 'test_matches.jsonl').open() as fin, open('test_matches.csv', 'w') as fout:
        process(fin, fout, limit=10000)

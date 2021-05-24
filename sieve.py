import re
import copy
import itertools as it
import operator as op

import numpy as np
import pandas as pd


def gen_sieve(state):
    captured = None
    while state is True or state.any():
        mask = yield captured
        captured = state & (True if mask is None else mask)
        state ^= captured
        if state is False:
            break
    yield captured


def sieve_stack(state, filters):
    if state is not True:
        state = state.copy()
    s = gen_sieve(state)
    next(s)
    res = {k: s.send(m) for k, m in filters}
    try:
        res['rem'] = s.send(None)
    except StopIteration:
        pass
    return res


def root(filters):
    return sieve_stack(True, filters)


def branch(filters, dct, *keys):
    d = copy.deepcopy(dct)
    d2 = d
    for k in iter(keys[:-1]):
        d2 = d2[k]

    d2[keys[-1]] = sieve_stack(d2[keys[-1]], filters)
    return d


def extend(filters, dct, *keys):
    d = copy.deepcopy(dct)
    d2 = d
    for k in iter(keys):
        d2 = d2[k]

    state = d2.pop('rem')
    d2.update(sieve_stack(state, filters))
    return d


def traverse(d, from_key=None):
    items = d.items()
    if from_key is not None:
        items = it.dropwhile(lambda kv: kv[0] != from_key, items)
    for k, m in items:
        if isinstance(m, dict):
            yield from traverse(m)
        else:
            yield k, m



x = pd.Series(range(10))

ma = root((
    (99, x>=9),
    (0, x<1),
    (1, x<2),
    ('odd', x % 2 == 1)
         ))

ma2 = branch((
       ('a', x<2),
       ('b', x<6)
), ma, 'odd')

ma3 = branch((
       (0, x<4),
), ma2, 'odd', 'b')

ma4 = extend((
    ('x', x < 3),
    ('y', x < 6)), ma3)

for k,m in traverse(ma4):
    print(k)
    print(x[m])

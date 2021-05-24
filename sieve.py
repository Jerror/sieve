import re
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


def branch(d, k, filters):
    d[k] = sieve_stack(d[k], filters)
    return d[k]



# def deepset(v, dct, *keys):
#     d = dct
#     for k in iter(keys[:-1]):
#         d = d[k]
#     d[keys[-1]] = v
#     return dct


def extend(d, filters):
    state = d.pop('rem')
    d.update(sieve_stack(state, filters))
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

branch(ma, 'odd', (
       ('a', x<2),
       ('b', x<6)
))

branch(ma['odd'], 'b', (
       (0, x<4),
))

extend(ma, (
    ('x', x < 3),
    ('y', x < 6)))

for k,m in traverse(ma):
    print(k)
    print(x[m])

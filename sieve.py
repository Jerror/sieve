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


class SieveTree:

    def __init__(self, filters):
        self.data = sieve_stack(True, filters)

    def branch(self, filters, *keys):
        new_tree = copy.deepcopy(self)
        d = new_tree.data
        for k in iter(keys[:-1]):
            d = d[k]

        d[keys[-1]] = sieve_stack(d[keys[-1]], filters)
        return new_tree

    def extend(self, filters, *keys):
        new_tree = copy.deepcopy(self)
        d = new_tree.data
        for k in iter(keys):
            d = d[k]

        state = d.pop('rem')
        d.update(sieve_stack(state, filters))
        return new_tree


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

st = SieveTree((
    (99, x>=9),
    (0, x<1),
    (1, x<2),
    ('odd', x % 2 == 1)
         ))

st2 = st.branch((
       ('a', x<2),
       ('b', x<6)
), 'odd')

st3 = st2.branch((
       (0, x<4),
), 'odd', 'b')

st4 = st3.extend((
    ('x', x < 3),
    ('y', x < 6)))

for t in [st, st2, st3, st4]:
    for k,m in traverse(t.data):
        print(k)
        print(x[m])
    print()

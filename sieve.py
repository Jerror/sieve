import re
import copy
import itertools as it
import operator as op

import numpy as np
import pandas as pd

from ete3 import Tree


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
    return filter(lambda kv: isinstance(kv[1], pd.Series) and kv[1].any(),
                  ((k, s.send(m)) for k, m in (*filters, ('rem', None))))


def root(filters):
    return dict(sieve_stack(True, filters))


def branch(d, k, filters):
    d[k] = dict(sieve_stack(d[k], filters))
    return d[k]


def extend(d, filters):
    state = d.pop('rem')
    d.update(sieve_stack(state, filters))
    return d


def recurse_items(d, from_key=None):
    items = d.items()
    if from_key is not None:
        items = it.dropwhile(lambda kv: kv[0] != from_key, items)
    for k, m in items:
        if isinstance(m, dict):
            yield from recurse_items(m)
        else:
            yield k, m


def dict2tree(d, prefix=''):
    root = Tree()
    n = root
    for k, v in d.items():
        n = n.add_child(name=k)
        if isinstance(v, dict):
            n.add_child(dict2tree(v, prefix=prefix + str(k) + ' '))
        else:
            n.add_child(name=prefix + str(k))
    n.delete()
    return root


class SieveTree:

    def __init__(self, filters):
        self.data = root(filters)

    def branch(self, filters, *keys, inplace=False):
        if inplace:
            new_tree = self
        else:
            new_tree = copy.deepcopy(self)
        d = new_tree.data
        for k in iter(keys[:-1]):
            d = d[k]

        branch(d, keys[-1], filters)
        return None if inplace else new_tree

    def extend(self, filters, *keys, inplace=False):
        if inplace:
            new_tree = self
        else:
            new_tree = copy.deepcopy(self)
        d = new_tree.data
        for k in iter(keys):
            d = d[k]

        extend(d, filters)
        return None if inplace else new_tree

    def traverse_leaves(self, *keys, from_key=None):
        d = self.data
        for k in iter(keys):
            d = d[k]

        return recurse_items(d, from_key)

    def get_tree(self):
        return dict2tree(self.data)


x = pd.Series(range(10))

st = SieveTree((
    (99, x >= 9),
    (0, x < 1),
    (1, x < 2),
    ('odd', x % 2 == 1),
))

st2 = st.branch((
    ('a', x < 2),
    ('b', x < 6),
), 'odd')

st3 = st2.branch(((0, x < 4), ), 'odd', 'b')

st4 = st3.extend((
    ('x', x < 3),
    ('y', x < 6),
))

for k, m in st4.traverse_leaves():
    print(k)
    print(x[m])
print()

print(st4.get_tree().get_ascii(show_internal=True))

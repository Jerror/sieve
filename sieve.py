import copy
import itertools as it
import pandas as pd
from numpy import ndarray
from ete3 import Tree
from dataclasses import dataclass
from typing import Union


# These types may contain valid "boolean vectors" for Pandas Series indexing
pandas_vec_types = (list, ndarray, pd.core.arrays.ExtensionArray, pd.Series, pd.Index)

@dataclass
class Leaf:
    data: Union[Union[pandas_vec_types], str]


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
        if not isinstance(state, pandas_vec_types):
            raise RuntimeError
        state = state.copy()
    s = gen_sieve(state)
    next(s)
    # append 'exhaustion' filter (None, None) and expand
    keys, masks = zip(*(*filters, (None, None)))
    # create leaves of sieved state, filtering invalid or empty results
    return filter(
        lambda km: isinstance(km[1].data, pandas_vec_types) and km[1].data.any(),
        zip(keys, map(lambda m: Leaf(s.send(m)), masks)))


def branch(d, k, filters):
    d[k] = dict(sieve_stack(d[k].data, filters))
    return d[k]


def extend(d, filters):
    state = d.pop(None).data
    d.update(sieve_stack(state, filters))
    return d


def recurse_items(d, *parents, from_key=None):
    items = d.items()
    if from_key is not None:
        items = it.dropwhile(lambda kv: kv[0] != from_key, items)
    for k, m in items:
        if isinstance(m, dict):
            yield from recurse_items(m, *parents, k)
        else:
            yield (k, m) if len(parents) == 0 else ((*parents, k), m)


def dict2tree(d, *parents):
    root = Tree()
    n = root
    for k, v in d.items():
        n = n.add_child(name=k)
        if isinstance(v, dict):
            n.add_child(dict2tree(v, *parents, k))
        elif isinstance(v.data, pandas_vec_types):
            label = str((*parents, k)) if v.data.any() else '()'
            n.add_child(name=label)
        else:
            n.add_child(name=str(v.data))
    n.delete()
    return root


class SieveTree:

    def __init__(self, state=True):
        self.d = {None: Leaf(state)}

    def branch(self, filters, *keys, inplace=False):
        if inplace:
            new_tree = self
        else:
            new_tree = copy.deepcopy(self)
        d = new_tree.d
        for k in iter(keys[:-1]):
            d = d[k]

        branch(d, keys[-1], filters)
        return None if inplace else new_tree

    def extend(self, filters, *keys, inplace=False):
        if inplace:
            new_tree = self
        else:
            new_tree = copy.deepcopy(self)
        d = new_tree.d
        for k in iter(keys):
            d = d[k]

        extend(d, filters)
        return None if inplace else new_tree

    def get_leaf(self, *keys):
        d = self.d
        for k in iter(keys[:-1]):
            d = d[k]

        return d[keys[-1]]

    def get(self, *keys):
        return self.get_leaf(*keys).data

    def traverse_leaves(self, *keys, from_key=None):
        d = self.d
        for k in iter(keys):
            d = d[k]

        return filter(
            lambda kv: isinstance(kv[1].data, pandas_vec_types) and kv[1].data.any(),
            recurse_items(d, from_key=from_key))

    def get_tree(self):
        return dict2tree(self.d)

    def __repr__(self):
        return self.get_tree().get_ascii(show_internal=True)


class Picker:

    def __init__(self, name, d):
        self.name = name
        self.d = d

    def pick_leaf(self, k, m):
        if not isinstance(m, Leaf):
            raise RuntimeError('Expected a Leaf')
        if not isinstance(m.data, pandas_vec_types):
            raise RuntimeError('Leaf already plucked: ' + str(m.data))

        if k in self.d:
            self.d[k] |= m.data
        else:
            self.d[k] = m.data
        m.data = self.name + ' ' + str(k)

    def pick_leaves(self, pairs):
        for k, m in pairs:
            self.pick_leaf(k, m)

    def merged(self):
        res = False
        for k, m in recurse_items(self.d):
            res |= m
        return res


def pretty_nested_dict_keys(d, indent=0):
    s = ''
    for key, value in d.items():
        s += '\t' * indent + str(key) + '\n'
        if isinstance(value, dict):
            s += pretty_nested_dict_keys(value, indent + 1)
    return s


class Results():

    def __init__(self):
        self.d = {}

    def picker(self, *keys):
        d = self.d
        for k in iter(keys):
            try:
                d = d[k]
            except KeyError:
                d[k] = {}
                d = d[k]
        return Picker(' '.join((str(k) for k in keys)), d)

    def __repr__(self):
        return pretty_nested_dict_keys(self.d)

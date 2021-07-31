from typing import Union, Callable
from collections.abc import Mapping
import copy
import itertools as it
import pandas as pd
from numpy import ndarray
from ete3 import Tree
from dataclasses import dataclass

# These types may contain valid "boolean vectors" for Pandas Series indexing
pandas_vec_types = (list, ndarray, pd.core.arrays.ExtensionArray, pd.Series,
                    pd.Index)


@dataclass
class Leaf:
    data: Union[Union[pandas_vec_types], str]


def gen_sieve(state):
    if state is None:
        state = True
    captured = None
    while state is True or state.any():
        mask = yield captured
        captured = state & (True if mask is None else mask)
        # not inplace: treat state parameter as immutable
        state = state ^ captured
        if state is False:
            break
    yield captured


def sieve_stack(state, filters, transform=None):
    if transform is None:
        transform = lambda x: x
    if state is not None:
        if not isinstance(state, pandas_vec_types):
            raise RuntimeError
    s = gen_sieve(state)
    next(s)
    # append 'exhaustion' filter (None, None) and expand
    keys, masks = zip(*(*filters, (None, None)))
    # create leaves of sieved state, filtering invalid or empty results
    return filter(
        lambda km: isinstance(km[1].data, pandas_vec_types) and km[1].data.any(
        ), zip(keys, map(lambda m: Leaf(s.send(transform(m))), masks)))


def recurse_items(d, *parents, from_key=None):
    items = d.items()
    if from_key is not None:
        items = it.dropwhile(lambda kv: kv[0] != from_key, items)
    for k, m in items:
        if isinstance(m, Mapping):
            yield from recurse_items(m, *parents, k)
        else:
            yield (k, m) if len(parents) == 0 else ((*parents, k), m)


class Sieve(Mapping):

    def __init__(self, state=None, transform=None):
        if transform is None:
            self.transform = lambda x: x
        elif transform == 'sparse':
            # Reduce memory complexity O(n^2) -> O(n) but lose packbits savings
            self.transform = lambda x: pd.arrays.SparseArray(
                x, dtype=bool, fill_value=False)
        elif isinstance(transform, Callable):
            self.transform = transform
        else:
            raise RuntimeError("Invalid transform")

        self.mapping = {None: Leaf(self._transform(state))}

    def _transform(self, x):
        if x is None:
            return x
        try:
            return self.transform(x)
        except Exception:
            return x

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, key):
        # Standard accessor with value typechecking.
        # Restricting vals to (Leaf, Sieve) enforces desired tree structure
        val = self.mapping[key]
        if not isinstance(val, (Leaf, Sieve)):
            raise LookupError("Invalid value type " + str(type(val)))
        return val

    def get_node(self, *keys):
        # Convenient nested access
        node = self
        for k in iter(keys[:-1]):
            node = node[k]
        return node[keys[-1]]

    def get_sieve(self, *keys):
        node = self.get_node(*keys)
        if not isinstance(node, Sieve):
            raise LookupError("Expected Leaf, got " + str(type(node)))
        return node

    def get_leaf(self, *keys):
        node = self.get_node(*keys)
        if not isinstance(node, Leaf):
            raise LookupError("Expected Leaf, got " + str(type(node)))
        return node

    def get_data(self, *keys):
        return self.get_leaf(*keys).data

    def extend(self, filters, *keys, inplace=False):
        sieve = self if inplace else copy.deepcopy(self)
        sub = sieve.get_sieve(*keys) if keys else sieve
        sub.mapping.update(
            sieve_stack(sub.mapping.pop(None).data, filters, self._transform))
        if not inplace:
            return sieve

    def branch(self, filters, *keys, inplace=False):
        sieve = self if inplace else copy.deepcopy(self)
        sub = sieve.get_sieve(keys[:-1]) if keys[:-1] else sieve
        sub.mapping[keys[-1]] = Sieve(sub.get_data(keys[-1]),
                                      transform=self.transform).extend(
                                          filters, inplace=False)
        if not inplace:
            return sieve

    def traverse_leaves(self, *keys, from_key=None):
        sieve = self.get_node(*keys) if keys else self
        return filter(
            lambda kv: isinstance(kv[1].data, pandas_vec_types) and kv[1].data.
            any(), recurse_items(sieve, from_key=from_key))

    def get_tree(self, *parents):
        root = Tree()
        n = root
        for k, v in self.items():
            n = n.add_child(name=k)
            if isinstance(v, Sieve):
                n.add_child(v.get_tree(*parents, k))
            elif isinstance(v.data, pandas_vec_types):
                label = str((*parents, k)) if v.data.any() else '()'
                n.add_child(name=label)
            else:
                n.add_child(name=str(v.data))
        n.delete()
        return root

    def __repr__(self):
        return self.get_tree().get_ascii(show_internal=True)


class Picker:

    def __init__(self, name, d):
        self.name = name
        self.mapping = d

    def pick_leaf(self, k, m):
        if not isinstance(m, Leaf):
            raise RuntimeError('Expected a Leaf')
        if not isinstance(m.data, pandas_vec_types):
            raise RuntimeError('Leaf already plucked: ' + str(m.data))

        if k in self.mapping:
            self.mapping[k] |= m.data
        else:
            self.mapping[k] = m.data
        m.data = self.name + ' ' + str(k)

    def pick_leaves(self, pairs):
        for k, m in pairs:
            self.pick_leaf(k, m)

    def merged(self):
        res = False
        for k, m in recurse_items(self.mapping):
            res |= m
        return res


def pretty_nested_dict_keys(d, indent=0):
    s = ''
    for key, value in d.items():
        s += '\t' * indent + str(key) + '\n'
        if isinstance(value, dict):
            s += pretty_nested_dict_keys(value, indent + 1)
    return s


class Results:

    def __init__(self):
        self.mapping = {}

    def picker(self, *keys):
        d = self.mapping
        for k in iter(keys):
            try:
                d = d[k]
            except KeyError:
                d[k] = {}
                d = d[k]
        return Picker(' '.join((str(k) for k in keys)), d)

    def __repr__(self):
        return pretty_nested_dict_keys(self.mapping)

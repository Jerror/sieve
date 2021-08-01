from typing import Union, Callable
from collections import UserDict
from collections.abc import Mapping
import copy
import itertools as it
import pandas as pd
from numpy import ndarray
from ete3 import Tree
from dataclasses import dataclass


def recurse_items(d, *parents, from_key=None):
    items = d.items()
    if from_key is not None:
        items = it.dropwhile(lambda kv: kv[0] != from_key, items)
    for k, m in items:
        if isinstance(m, Mapping):
            yield from recurse_items(m, *parents, k)
        else:
            yield (k, m) if len(parents) == 0 else ((*parents, k), m)


@dataclass
class Leaf:
    data: Union[pd.DataFrame, str]


class Sieve(Mapping):

    def __init__(self, state):
        self.mapping = {None: Leaf(state)}

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

        state = sub.mapping.pop(None).data
        for k, f in (*filters, (None, None)):
            if state.empty:
                break
            if k in sub:
                raise RuntimeError("Key " + str(k) + " is not unique")
            if f is None:
                sub.mapping[k] = Leaf(state)
                state = pd.DataFrame()
            else:
                if isinstance(f, Callable):
                    mask = f(state)
                else:
                    mask = state.eval(f)
                if mask.any():
                    sub.mapping[k] = Leaf(state[mask])
                    state = state[~mask]

        if not inplace:
            return sieve

    def branch(self, filters, *keys, inplace=False):
        sieve = self if inplace else copy.deepcopy(self)
        sub = sieve.get_sieve(keys[:-1]) if keys[:-1] else sieve

        sub.mapping[keys[-1]] = Sieve(sub.get_data(keys[-1])).extend(
            filters, inplace=False)
        if not inplace:
            return sieve

    def traverse_leaves(self, *keys, from_key=None):
        sieve = self.get_node(*keys) if keys else self
        return filter(
            lambda kv: isinstance(kv[1].data, pd.DataFrame) and not kv[1].data.
            empty, recurse_items(sieve, from_key=from_key))

    def get_tree(self, *parents):
        root = Tree()
        n = root
        for k, v in self.items():
            n = n.add_child(name=k)
            if isinstance(v, Sieve):
                n.add_child(v.get_tree(*parents, k))
            elif isinstance(v.data, pd.DataFrame):
                label = str((*parents, k)) if not v.data.empty else '()'
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
        if not isinstance(m.data, pd.DataFrame):
            raise RuntimeError('Leaf already plucked: ' + str(m.data))

        if k in self.mapping:
            self.mapping[k] = pd.concat((self.mapping[k], m.dataframes),
                                        axis=1,
                                        copy=False)
        else:
            self.mapping[k] = m.data
        m.data = self.name + ' ' + str(k)

    def pick_leaves(self, pairs):
        for k, m in pairs:
            self.pick_leaf(k, m)

    def merged(self):
        if self.mapping:
            _, frames = zip(*recurse_items(self.mapping))
            return pd.concat(frames, axis=1, copy=False)
        else:
            return pd.DataFrame()


def pretty_nested_dict_keys(d, indent=0):
    s = ''
    for key, value in d.items():
        s += '\t' * indent + str(key) + '\n'
        if isinstance(value, Mapping):
            s += pretty_nested_dict_keys(value, indent + 1)
    return s


class Results(UserDict):

    def picker(self, *keys):
        d = self.data
        for k in iter(keys):
            try:
                d = d[k]
            except KeyError:
                d[k] = Results()
                d = d[k]
        return Picker(' '.join((str(k) for k in keys)), d)

    def __repr__(self):
        return pretty_nested_dict_keys(self.data)

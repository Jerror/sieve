import copy
import itertools as it
import subprocess
from tempfile import NamedTemporaryFile
from collections import UserDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Union

import pandas as pd
from ete3 import Tree


def fun_contains_str(col, patt, **kwargs):
    return lambda df, patt=patt, kwargs=kwargs: df[col].str.contains(
        patt, **kwargs)


def fun_date_isin(col, dates):
    return lambda df, dates=dates: df[col].dt.tz_localize(None).astype(
        "datetime64[D]").isin(dates)


def reduce_matching(df, matchcol, sumcols=None, match=None, fillna=None):
    mycol = df[matchcol]
    if fillna is not None:
        if fillna in mycol.values:
            print('Warning: found ' + str(fillna) + ' in preexisting values')
        mycol = mycol.fillna(fillna)
    if match is None:
        match = mycol.unique()
    if sumcols is None:
        return match
    dat = pd.concat([
        pd.DataFrame(df[mycol == d][sumcols].sum()).transpose() for d in match
    ],
                    join="inner",
                    ignore_index=True)
    dat.insert(len(sumcols), matchcol, match)
    dat.sort_values(by=sumcols, ascending=False, inplace=True)
    return dat


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


class SieveTree(Mapping):

    def __init__(self, state):
        self.mapping = {None: Leaf(state)}

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, key):
        # Standard accessor with value typechecking.
        # Restricting vals to (Leaf, SieveTree) enforces desired tree structure
        val = self.mapping[key]
        if not isinstance(val, (Leaf, SieveTree)):
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
        if not isinstance(node, SieveTree):
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

        sub.mapping[keys[-1]] = SieveTree(sub.get_data(keys[-1])).extend(
            filters, inplace=False)
        if not inplace:
            return sieve

    def reduce_remainder(self,
                         matchcol,
                         sumcols=None,
                         match=None,
                         fillna=None):
        return reduce_matching(self.get_data(None), matchcol, sumcols, match,
                               fillna)

    def traverse_leaves(self, *keys, from_key=None):
        sieve = self.get_node(*keys) if keys else self
        return filter(
            lambda kv: isinstance(kv[1].data, pd.DataFrame) and not kv[1].data.
            empty, recurse_items(sieve, from_key=from_key))

    def traverse_data(self, *keys, from_key=None):
        return ((k, v.data)
                for k, v in self.traverse_leaves(*keys, from_key=None))

    def get_tree(self, *parents):
        root = Tree()
        n = root
        for k, v in self.items():
            n = n.add_child(name=k)
            if isinstance(v, SieveTree):
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

    def table(self, *keys, path=None, align=True, table_right=None, **kwargs):
        sep = "\xFE"
        out = ''
        first = True
        for k, v in self.traverse_leaves(*keys, **kwargs):
            if first:
                header = [
                    'index' if x is None else x
                    for x in v.data.index.names + list(v.data.columns)
                ]
                out += sep.join(header) + '\n'
            # With padding to cheat column -E
            out += 100 * '#' + str((k, ))[1:-2] + '\n'
            out += v.data.to_csv(None, sep=sep, header=False)
            first = False

        if align:
            N = ' -N' + ','.join(header) + ' '
            E = ' -E' + ','.join([header[0], header[-1]]) + ' '
            R = ''
            if table_right is not None:
                R = ' -R' + ','.join(table_right) + ' '

            out = subprocess.check_output(
                "sed '/^#/!s/,/" + sep + "/g' | column -t -d" + N + E + R +
                "-s" + sep + " | sed '/^\\s*$/d' | sed 's/^#\\+/# /'",
                input=out,
                shell=True,
                encoding='utf-8')

        if path is None:
            return out
        else:
            with open(path, 'w') as f:
                f.write(out)

    def diff(self, other, context=3):
        with NamedTemporaryFile('w') as f:
            f.write(other.table())
            f.flush()
            with NamedTemporaryFile('w') as f2:
                f2.write(self.table())
                f2.flush()
                diff = subprocess.run('diff --show-function-line="^#" -U ' +
                                      str(context) + ' ' + f.name + ' ' +
                                      f2.name,
                                      shell=True,
                                      stdout=subprocess.PIPE).stdout
        return diff.decode()

    def copy(self):
        return copy.deepcopy(self)


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
            self.mapping[k] = pd.concat((self.mapping[k], m.data),
                                        axis=0,
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
            return pd.concat(frames, axis=0, copy=False)
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


def varargs_comp(y, *x):
    for yy, xx in zip(y, iter(x)):
        if yy != xx:
            return False
    return True


class Sieve:

    def __init__(self, state):
        self.tree = SieveTree(state)
        self.results = Results()

    def get_results(self, *keys):
        # Convenient nested access
        res = self.results
        for k in iter(keys[:-1]):
            res = res[k]
        return res[keys[-1]]

    def extend(self, filters, *keys):
        filters = list(filters)
        self.tree.extend(filters, *keys, inplace=True)
        return self.tree.table(*keys, from_key=filters[0][0])

    def branch(self, filters, *keys):
        self.tree.branch(filters, *keys, inplace=True)
        return self.tree.table(*keys)

    def pick(self, pickkeys_list, *reskeys):
        for pickkeys in pickkeys_list:
            self.results.picker(*reskeys).pick_leaf(
                pickkeys[0], self.tree.get_leaf(*pickkeys[1:]))

    def merge(self, *keys):
        res = self.get_results(*keys[:-1]) if keys[:-1] else self.results
        res[keys[-1]] = self.results.picker().merged()

    def find_keys(self, match):
        if match is None:
            return (None, ) if None in self.tree else tuple()
        it = (k for k, _ in self.tree.traverse_leaves())
        return tuple(
            filter(lambda k: varargs_comp(match, *k),
                   filter(lambda k: k is not None, it)))

    def __repr__(self):
        return self.results.__repr__()

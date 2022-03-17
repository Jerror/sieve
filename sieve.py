import re
import copy
import itertools as it
import subprocess
from tempfile import NamedTemporaryFile
from collections import UserDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable, Union

from numpy import ndarray
import pandas as pd
from ete3 import Tree

# These types may contain valid "boolean vectors" for Pandas Series indexing
pandas_vec_types = (list, ndarray, pd.core.arrays.ExtensionArray, pd.Series,
                    pd.Index)


def fun_contains_str(col, patt, **kwargs):
    """ Return callback for pandas.Series.str.contains filtering"""

    # Some default kwargs which I use often
    myargs = {'na': False}
    if not isinstance(patt, re.Pattern):
        myargs.update({'case': False})
    # **kwargs can overwrite the defaults
    myargs.update(kwargs)
    return lambda df: df[col].str.contains(patt, **myargs)


def fun_date_isin(datecol, dates):
    """ Return callback for filtering a list of dates in datecol """

    return lambda df: df[datecol].dt.tz_localize(None).astype("datetime64[D]"
                                                              ).isin(dates)


def reduce_matching(df, matchcol, sumcols=None, match=None):
    """ Reduce df by summation over sumcols of rows where matchcol matches
    values in match. If match is None, reduce on unique values in matchcol.
    """

    mycol = df[matchcol]
    if match is None:
        match = mycol.unique()
    if sumcols is not None and not isinstance(sumcols, list):
        sumcols = [sumcols]

    rows = []
    for d in match:
        matching = df[mycol == d if not pd.isna(d) else mycol.isna()]
        if sumcols is not None:
            row = pd.DataFrame(matching[sumcols].sum()).transpose()
            row.insert(0, 'Count', len(matching))
        else:
            row = pd.DataFrame({'Count': [len(matching)]})
        rows.append(row)

    dat = pd.concat(rows, join="inner", ignore_index=True)
    dat.insert(0, matchcol, match)
    return dat.sort_values(by='Count', ascending=False).reset_index(drop=True)


def recurse_mapping(d, *parents):
    """ Return depth-first iterator over mapping which traverses nested
    mappings as encountered. Items are (keys, value) pairs where keys
    is a tuple including the keys of parent mappings. """

    for k, m in d.items():
        if isinstance(m, Mapping):
            yield from recurse_mapping(m, *parents, k)
        else:
            yield ((*parents, k), m)


def select_items(items, from_key=None, to_key=None, key_filter=None):
    """ Return iterator over (key, value) items from key from_key until key
    to_key where function key_filter maps keys to True """

    if from_key is not None:
        items = it.dropwhile(lambda kv: kv[0] != from_key, items)
    if to_key is not None:
        items = it.takewhile(lambda kv: kv[0] != to_key, items)
    if key_filter is not None:
        items = filter(lambda kv: key_filter(kv[0]), items)
    return items


def table(objs, **kwargs):
    """ Return table of dataframes with data from each
    frame headed by #<keys specifying leaf>.
    """

    combined_df = pd.concat({'# ' + str(k) + '\xFE': df for k, df in objs})
    table = combined_df.to_string(**kwargs)
    # Put keys on their own line
    table = table.replace('\xFE', '\n')
    table_width = len(table.splitlines()[-1])
    # Remove leading whitespace from rows
    table = re.sub('\n\\s+', '\n', table)
    delta = table_width - len(table.splitlines()[-1])
    # Realign header
    table = table[delta:]
    return table


def diff_tables(table1, table2, context=0, labels=None):
    """ Return the unified difference of two tables. context specifies
    the number of lines of context printed about differences.
    The system diff utility is used instead of difflib because the latter
    lacks the functionality of --show-function-line, which we use here
    to highlight under which '# <heading>' a diff occurred. """

    if labels is None:
        labels = ['TABLE 1', 'TABLE 2']

    with NamedTemporaryFile('w') as f:
        f.write(table2)
        f.flush()
        with NamedTemporaryFile('w') as f2:
            f2.write(table1)
            f2.flush()
            diff = subprocess.run(
                'diff -b --show-function-line="^#" ' + f'-U {context} ' +
                ' --label "{1}" --label "{0}" '.format(*labels) + f.name +
                ' ' + f2.name,
                shell=True,
                stdout=subprocess.PIPE).stdout
            lines = diff.decode().splitlines()
    return '\n'.join(lines[:2] + [' ' + table1.splitlines()[0]] + lines[2:])


@dataclass
class Leaf:
    """ Dataclass for leaves of SieveTree. Contains either a DataFrame which
    is a partition produced by the SieveTree filter cascade, or a string;
    in intended usage string data indicates that the partition has been
    gathered into a Results object by a Picker, and the string names the keys
    locating the data in the results. """
    data: Union[pd.DataFrame, str]


class SieveTree(Mapping):
    """ A Mapping with methods for partitioning a DataFrame via sequences of
    filters and storing each partition mapped to a tuple of keys. Each filter
    divides a partition into a captured part and a remainder, each of which can
    be filtered further. The resulting datastructure is a full binary tree
    whose leaves contain the DataFrame partitions and whose internal nodes
    correspond to the application of a filter. Filters can only be added
    downstream.

    The mental model of the design is that of a branching stack of sieves,
    where each sieve in a stack acts to filter that part of the data which
    was not captured by the sieve above. A sequence of filters passed to the
    branch or extend methods each act in turn on that which was *not* captured
    by the previous filter. The extend method adds more filters to the bottom
    of a specified stack. The branch method creates a new stack to refine the
    data which was captured by a specified filter.

    Each filter in sequence is specified by a (key, action) pair. The keys must
    be unique within a stack. They label the corresponding nodes of the tree
    and are used to access branches and leaves: a position in the tree is given
    by the sequence of keys corresponding to the sequence of filters which
    captured data en route to the state. The special key None is reserved for
    the data which remains uncaptured at the bottom of a stack.

    The data contained in a leaf can be either a DataFrame partition or a string.
    If data is a string the leaf is passed over in tree traversal. Typically
    string data means that the partition was moved to some mapping via Picker,
    and the string names the location of the partition in the mapping."""

    def __init__(self, state):
        self.mapping = {None: Leaf(state)}

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, key):
        # Standard accessor with value typechecking.
        val = self.mapping[key]
        # Restricting vals to (Leaf, SieveTree) enforces desired tree structure
        if not isinstance(val, (Leaf, SieveTree)):
            raise LookupError("Invalid value type " + str(type(val)))
        return val

    def get_node(self, *keys):
        # Convenient nested access
        if keys:
            node = self
            for k in iter(keys[:-1]):
                node = node[k]
            return node[keys[-1]]
        else:
            return self

    def get_sieve(self, *keys):
        # Get subtree with nested accessing and type checking
        node = self.get_node(*keys)
        if not isinstance(node, SieveTree):
            raise LookupError("Expected SieveTree, got " + str(type(node)))
        return node

    def get_leaf(self, *keys):
        # Get leaf with nested accessing and type checking
        node = self.get_node(*keys)
        if not isinstance(node, Leaf):
            raise LookupError("Expected Leaf, got " + str(type(node)))
        return node

    def get_data(self, *keys):
        # Get data from leaf with nested accessing and type checking
        return self.get_leaf(*keys).data

    def extend(self, filters, *keys, inplace=False):
        """ Extend branch specified by *keys with nodes created
        from (iterable) filters. Filters act on the state which *doesn't*
        match the previous filter, starting from the remainder leaf with
        special key None of the specified branch; any leftover state is left
        in a new None leaf. If state is exhausted filtration is ceased and
        if a filter finds no matches no corresponding node is created.

        Each filter is a (key, value) pair where key becomes the key of the
        node in the branch mapping and value specifies the filtration. If
        value is Callable it is called on the state. If value is a string it
        is passed to state.eval. In any case the result is assumed to be a
        boolean vector. """

        sieve = self if inplace else copy.deepcopy(self)
        sub = sieve.get_sieve(*keys)

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
                elif isinstance(f, str):
                    mask = state.eval(f)
                else:
                    raise TypeError(
                        "Expected Callable or string filter, got " +
                        str(type(f)))
                if not isinstance(mask, pandas_vec_types):
                    raise TypeError(
                        "Filter must return a boolean vector, got " +
                        str(type(mask)))
                if mask.any():
                    sub.mapping[k] = Leaf(state[mask])
                    state = state[~mask]

        if not inplace:
            return sieve

    def branch(self, filters, *keys, inplace=False):
        """ Create a new branch on the state captured at leaf specified by
        *keys and extend with filters. """

        sieve = self if inplace else copy.deepcopy(self)
        sub = sieve.get_sieve(*keys[:-1])

        sub.mapping[keys[-1]] = SieveTree(sub.get_data(keys[-1])).extend(
            filters, inplace=False)
        if not inplace:
            return sieve

    def traverse_leaves(self, *keys, **kwargs):
        """ Return depth-first clockwise iterator over leaves from branch at
        *keys filtered by select_items according to **kwargs and skipping
        leaves containing data which is not a non-empty DataFrame. Items are
        (keys, leaf) pairs where keys is a tuple including all parent keys. """

        items = (((*keys, *k), v)
                 for k, v in select_items(recurse_mapping(self.get_sieve(*keys)), **kwargs))
        return filter(
            lambda kv: isinstance(kv[1].data, pd.DataFrame) and not kv[1].data.
            empty, items)

    def traverse_data(self, *keys, **kwargs):
        # Same as traverse_leaves, but items contain leaf data instead of leaf
        return ((k, v.data) for k, v in self.traverse_leaves(*keys, **kwargs))

    def traverse_keys(self, *keys, **kwargs):
        # Same as traverse_leaves but items only contain the keys
        return (k for k, _ in self.traverse_leaves(*keys, **kwargs))

    def get_tree(self, *parents):
        # Get ete3 Tree representation of SieveTree structure
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
        # ASCII representation of SieveTree structure
        return self.get_tree().get_ascii(show_internal=True)

    def dataframe(self, *keys, **kwargs):
        """ Return a DataFrame combining all data in tree beneath specified
        node with multiindex specifying data leaf keys. """

        return pd.concat(
            {str(k): d
             for k, d in self.traverse_data(*keys, **kwargs)})

    def table(self, *keys, formatting=None, **kwargs):
        """ Return table of tree data in traverse_leaves
        order with data from a given leaf headed by #<keys specifying leaf>.
        """

        if formatting is None:
            formatting = {}
        return table(self.traverse_data(*keys, **kwargs), **formatting)

    def diff(self, other, *keys, context=0, **kwargs):
        """ Diff the table of this SieveTree with another. context specifies
        the number of lines of context printed about differences.
        This is mainly meant to be used to determine the trickle-down effects
        of changing filters 'upstream' by duplicating and modifying the tree
        creation code. """

        return diff_tables(self.table(*keys, **kwargs),
                           other.table(*keys, **kwargs),
                           context=context,
                           labels=['self', 'other'])

    def copy(self):
        return copy.deepcopy(self)


class Picker:
    """ A helper class for moving data from SieveTree leaves into another
    mapping. Handles data merging and replacement of leaf data with a
    destination string. """

    def __init__(self, name, d):
        self.name = name
        self.mapping = d

    def pick_leaf(self, k, m):
        """ Put data from a leaf m into mapping at key k. If data exists at
        key k, concatenates the data. Replaces leaf.data with string
        self.name+k. """

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

    def merged(self):
        """ Concatenate all data in self.mapping, recursing on nested
        mappings. """

        if self.mapping:
            _, frames = zip(*recurse_mapping(self.mapping))
            return pd.concat(frames, axis=0, copy=False)
        else:
            return pd.DataFrame()


def pretty_nested_dict_keys(d, indent=0):
    """ Return a string listing all keys in a Mapping and any
    sub-Mappings, separated by line breaks with indentation level
    corresponding to sub-Mapping depth. """
    s = ''
    for key, value in d.items():
        s += '\t' * indent + str(key) + '\n'
        if isinstance(value, Mapping):
            s += pretty_nested_dict_keys(value, indent + 1)
    return s


class Results(UserDict):
    """ Dictionary for results collected from SieveTree leaves. Used to select,
    categorize and recombine leaf data. The picker method automatically creates
    nested Results as required and returns an appropriate Picker object. """

    def picker(self, *keys):
        """ Return a Picker whose mapping is the Results object at
        self.data[keys[0]][keys[1]]... with name given by concatenation
        of keys. Nested Results objects are created as necessary. """

        d = self.data
        for k in iter(keys):
            try:
                d = d[k]
            except KeyError:
                d[k] = Results()
                d = d[k]
        return Picker(' '.join((str(k) for k in keys)), d)

    def get_node(self, *keys):
        # Convenient nested access
        if keys:
            res = self
            for k in iter(keys[:-1]):
                res = res[k]
            return res[keys[-1]]
        else:
            return self

    def get_results(self, *keys):
        # Get sub-results with nested accessing and type checking
        node = self.get_node(*keys)
        if not isinstance(node, Results):
            raise LookupError("Expected Results, got " + str(type(node)))
        return node

    def get_data(self, *keys):
        # Get data with nested accessing and type checking
        node = self.get_node(*keys)
        if not isinstance(node, pd.DataFrame):
            raise LookupError("Expected DataFrame, got " + str(type(node)))
        return node

    def apply(self, fun, *keys):
        """ Replace value at *keys with the result of applying callback fun to
        the value. """

        res = self.get_results(*keys[:-1])
        res[keys[-1]] = fun(res[keys[-1]])

    def merge(self, *keys):
        """ Replace Results object at *keys with its recursively concatenated
        data. """

        self.apply(lambda res: res.picker().merged(), *keys)

    def traverse_data(self, *keys, **kwargs):
        """ Return iterator over (keys, df) pairs in top-down order
        starting from from_key in Results at *keys. """

        items = (((*keys, *k), v)
                 for k, v in select_items(recurse_mapping(self.get_results(*keys)), **kwargs))
        return filter(
            lambda kv: isinstance(kv[1], pd.DataFrame) and not kv[1].empty,
            items)

    def traverse_keys(self, *keys, **kwargs):
        # Same as traverse_data but items only contain the keys
        return (k for k, _ in self.traverse_data(*keys, **kwargs))

    def dataframe(self, *keys, **kwargs):
        """ Return a DataFrame combining all data in Results beneath specified
        node with multiindex specifying the data's sequence of keys. """

        return pd.concat(
            {str(k): d
             for k, d in self.traverse_data(*keys, **kwargs)})

    def table(self, *keys, formatting=None, **kwargs):
        """ Return table of data in traverse_data order with each block of data
        headed by #<keys locating data>. """

        if formatting is None:
            formatting = {}
        return table(self.traverse_data(*keys, **kwargs), **formatting)

    def diff(self, other, *keys, context=0, **kwargs):
        """ Diff the table of this Results with another. context specifies
        the number of lines of context printed about differences.
        This is mainly meant to be used to determine the trickle-down effects
        of changing filters 'upstream' by duplicating and modifying the tree
        creation code. """

        return diff_tables(self.table(*keys, **kwargs),
                           other.table(*keys, **kwargs),
                           context=context,
                           labels=['self', 'other'])

    def __repr__(self):
        return pretty_nested_dict_keys(self.data)

    def copy(self):
        return copy.deepcopy(self)


def is_iterable(arg):
    return isinstance(arg, Iterable) and not isinstance(arg, str)


class Sieve:
    """ Combines SieveTree, Results and Picker into one simple object intended
    for practical use. A SieveTree (self.tree) and a corresponding Results
    (self.results) object are member variables; the pick method conveniently
    populates results from tree leaves given the keys specifying location in
    results and the leaves to pick, so it is not necessary to work with Picker
    objects directly. The extend and branch methods modify the tree in-place.
    Methods extend, branch and pick each take a dry_run argument; if true then
    data is not modified and a DataFrame describing the additions that would
    have been made is returned. """

    def __init__(self, state):
        self.tree = SieveTree(state)
        self.results = Results()

    def extend(self, filters, *keys, dry_run=False):
        """ Extend self.tree (see SieveTree.extend) in-place if dry_run is
        False, otherwise return a DataFrame describing the extension that
        would have been created. """

        if dry_run:
            filters = list(filters)
            tree = self.tree.extend(filters, *keys, inplace=False)
            return tree.dataframe(*keys, from_key=filters[0][0])
        else:
            self.tree.extend(filters, *keys, inplace=True)

    def branch(self, filters, *keys, dry_run=False):
        """ Branch self.tree (see SieveTree.branch) in-place if dry_run is
        False, otherwise return a DataFrame describing the branch that would
        have been created. """

        if dry_run:
            tree = self.tree.branch(filters, *keys, inplace=False)
            return tree.dataframe(*keys)
        else:
            self.tree.branch(filters, *keys, inplace=True)

    def pick(self, pickkeys_list, *reskeys, dry_run=False, skip_missing=False):
        """ Pluck leaves from self.tree to Results object in self.results at
        (nested) key(s) *reskeys. Each item in pickkeys_list is a tuple of keys,
        where the first key names the entry in the Results object and the
        remaining keys specify a leaf in self.tree. If dry_run is True instead
        return a DataFrame of data that would have been picked with index naming
        the intended location in results. """

        # For items which are a flat iterable of keys (assumed to be leaf keys)
        #  set results key to last leaf key
        pickkeys_list = (pk if is_iterable(pk[-1]) else (pk[-1], pk)
                         for pk in pickkeys_list)

        key_leaf_list = []
        for pk in pickkeys_list:
            try:
                leaf = self.tree.get_leaf(*pk[1])
            except KeyError as e:
                if skip_missing:
                    continue
                else:
                    raise e
            key_leaf_list += [(pk[0], leaf)]

        if dry_run:
            return pd.concat({
                str((*reskeys, key)): leaf.data
                for key, leaf in key_leaf_list
            })
        else:
            picker = self.results.picker(*reskeys)
            for key, leaf in key_leaf_list:
                picker.pick_leaf(key, leaf)

    def __repr__(self):
        return self.tree.__repr__() + '\n\n' + self.results.__repr__()

    def copy(self):
        return copy.deepcopy(self)

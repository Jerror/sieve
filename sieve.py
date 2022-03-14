import re
import copy
import itertools as it
import subprocess
from tempfile import NamedTemporaryFile
from collections import UserDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Union

from numpy import ndarray
import pandas as pd
from ete3 import Tree

# These types may contain valid "boolean vectors" for Pandas Series indexing
pandas_vec_types = (list, ndarray, pd.core.arrays.ExtensionArray, pd.Series,
                    pd.Index)


def fun_contains_str(col, patt, **kwargs):
    # Return callback for pandas.Series.str.contains filtering
    return lambda df, patt=patt, kwargs=kwargs: df[col].str.contains(
        patt, **kwargs)


def fun_date_isin(datecol, dates):
    # Return callback for date filtering
    return lambda df, dates=dates: df[datecol].dt.tz_localize(None).astype(
        "datetime64[D]").isin(dates)


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


def recurse_items(d, *parents, from_key=None):
    """ Return depth-first iterator over mapping which traverses nested
    mappings as encountered. Items are (keys, value) pairs where keys
    is a tuple including the keys of parent mappings. """

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
    the data which remains uncaptured at the bottom of a stack. """

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
        node = self
        for k in iter(keys[:-1]):
            node = node[k]
        return node[keys[-1]]

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
        sub = sieve.get_sieve(keys[:-1]) if keys[:-1] else sieve

        sub.mapping[keys[-1]] = SieveTree(sub.get_data(keys[-1])).extend(
            filters, inplace=False)
        if not inplace:
            return sieve

    def reduce_remainder(self, matchcol, *keys, sumcols=None, match=None):
        # Perform reduction on the remainder state of branch at *keys
        sub = self.get_sieve(*keys) if keys else self
        return reduce_matching(sub.get_data(None),
                               matchcol,
                               sumcols=sumcols,
                               match=match)

    def traverse_leaves(self, *keys, from_key=None):
        """ Return depth-first iterator over leaves from branch at *keys starting
        at from_key which traverses branches as encountered and skips leaves
        containing data which is not a non-empty DataFrame. Items are (keys,
        leaf) pairs where keys is a tuple including all parent keys.
        """

        sieve = self.get_node(*keys) if keys else self
        return filter(
            lambda kv: isinstance(kv[1].data, pd.DataFrame) and not kv[1].data.
            empty, recurse_items(sieve, *keys, from_key=from_key))

    def traverse_data(self, *keys, **kwargs):
        # Same as traverse_leaves, but items contain leaf data instead of leaf
        return ((k, v.data) for k, v in self.traverse_leaves(*keys, **kwargs))

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

    def dataframe(self, *keys, from_key=None):
        ks, dfs = zip(*self.traverse_data(*keys, from_key=from_key))
        return pd.concat(dfs, keys=(str((k, ))[1:-2] for k in ks))

    def table(self, *keys, from_key=None, path=None, **kwargs):
        ks, dfs = zip(*self.traverse_data(*keys, from_key=from_key))
        combined_df = pd.concat(
            dfs, keys=['# ' + str((k, ))[1:-2] + '\xFE' for k in ks])
        table = combined_df.to_string(**kwargs)
        # Put keys on their own line
        table = table.replace('\xFE', '\n')
        table_width = len(table.splitlines()[-1])
        # Remove leading whitespace from rows
        table = re.sub('\n\\s+', '\n', table)
        delta = table_width - len(table.splitlines()[-1])
        # Realign header
        table = table[delta:]

        if path is None:
            return table
        else:
            with open(path, 'w') as f:
                f.write(table)

    def diff(self, other, *keys, context=0, **kwargs):
        """ Diff the table of this SieveTree with another. context specifies
        the number of lines of context printed about differences.
        This is mainly meant to be used to determine the trickle-down effects
        of changing filters 'upstream' by duplicating and modifying the tree
        creation code. """

        with NamedTemporaryFile('w', prefix='OTHER_') as f:
            f.write(other.table(*keys, **kwargs))
            f.flush()
            with NamedTemporaryFile('w', prefix='SELF_') as f2:
                f2.write(self.table(*keys, **kwargs))
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

    def pick_leaves(self, pairs):
        # Pick multiple leaves in one call. For convenience
        for k, m in pairs:
            self.pick_leaf(k, m)

    def merged(self):
        """ Concatenate all data in self.mapping, recursing on nested
        mappings. """
        if self.mapping:
            _, frames = zip(*recurse_items(self.mapping))
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

    def __repr__(self):
        return pretty_nested_dict_keys(self.data)


def varargs_comp(y, *x):
    # Check if N args *x are equal to the first N items of iterable y
    for yy, xx in zip(y, iter(x)):
        if yy != xx:
            return False
    return True


class Sieve:
    """ Combines SieveTree, Results and Picker into one simple object intended
    for practical use. A SieveTree (self.tree) and a corresponding Results
    (self.results) object are member variables; the pick method conveniently
    populates results from tree leaves given the keys specifying location in
    results and the leaves to pick, so it is not necessary to work with Picker
    objects directly. The extend and branch methods modify the tree in-place
    unless kwarg dry_run is true in which case they do not modify anything but
    print the tree and a table showing the would-be effects of the call. A
    dictionary of kwargs for tree.table formatting can be provided in
    initialization."""

    def __init__(self, state):
        self.tree = SieveTree(state)
        self.results = Results()

    def get_results(self, *keys):
        # Convenient nested access
        res = self.results
        for k in iter(keys[:-1]):
            res = res[k]
        return res[keys[-1]]

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

    def pick(self, pickkeys_list, *reskeys):
        """ Pluck leaves from self.tree to Results object in self.results at
        (nested) key(s) *reskeys. Each item in pickkeys_list is a tuple of keys,
        where the first key names the entry in the Results object and the
        remaining keys specify a leaf in self.tree. """
        picker = self.results.picker(*reskeys)
        for pickkeys in pickkeys_list:
            picker.pick_leaf(pickkeys[0], self.tree.get_leaf(*pickkeys[1:]))

    def merge(self, *keys):
        """ Replace Results object at *keys with its recursively concatenated
        data. """
        res = self.get_results(*keys[:-1]) if keys[:-1] else self.results
        res[keys[-1]] = self.results.picker().merged()

    def find_keys(self, *match):
        """ Find all tuples of keys specifying unplucked leaves where the first
        len(match) keys in the tuple are itemwise equal to match """
        if match == (None, ):
            return (None, ) if None in self.tree else tuple()
        it = (k for k, _ in self.tree.traverse_leaves())
        return tuple(
            filter(lambda k: varargs_comp(match, *k),
                   filter(lambda k: k is not None, it)))

    def __repr__(self):
        return self.tree.__repr__() + '\n\n' + self.results.__repr__()

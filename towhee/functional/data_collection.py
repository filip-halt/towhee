# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterable, Iterator, Callable
import reprlib

from towhee.functional.mixins.dag import register_dag
from towhee.hparam import param_scope, dynamic_dispatch
from towhee.functional.option import Option, Some
from towhee.functional.entity import EntityView
from towhee.functional.mixins import DCMixins
from towhee.functional.mixins.dataframe import DataFrameMixin
from towhee.functional.mixins.column import ColumnMixin


class DataCollection(Iterable, DCMixins):
    """A pythonic computation and processing frameowrk.

    DataCollection is a pythonic computation and processing framework for unstructured
    data in machine learning and data science. It allows a data scientist or researcher
    to assemble data processing pipelines and do their model work (embedding,
    transforming, or classification) with a method-chaining style API. It is also
    designed to behave as a python list or iterator.

    Cases:
        1. `DataCollection(iter([1, 2, 3])).stage1().stage2()`

        When a `DataCollection` object is created from an iterator, it behaves as a
        python iterator and performs `stream-wise` data processing::

            a. `DataCollection` takes one element from the input and applies `stage1`,
                output is fed to a new `DataCollection` where `stage2` is sequentially
                applied.

            b. Functions such as indexing or shuffle are not supported due to no data
                being stored in the DataCollection.

        2. `DataCollection([1, 2, 3]).stage1().stage2()`

        When a `DataCollection` object is created from a list, it will hold all the
        input values, and perform stage-wise computations::

            a. `stage2` will wait until all the calculations are done in `stage1`.

            b. A new DataCollection will be created to hold all the outputs for each
                stage. You can perform list operations on result DataCollection.

    Examples:
        1. Create a DataCollection from list or iterator::

        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc = DataCollection(iter([0, 1, 2, 3, 4]))

        2. Chaining function invocations makes your code clean and fluent::

        >>> (
        ...    dc.map(lambda x: x+1)
        ...      .map(lambda x: x*2)
        ... ).to_list()
        [2, 4, 6, 8, 10]

        3. Multi-line closures are also supported via decorator syntax::

        >>> dc = DataCollection([1,2,3,4])
        >>> @dc.map
        ... def add1(x):
        ...     return x+1
        >>> @add1.map
        ... def mul2(x):
        ...     return x *2
        >>> @mul2.filter
        ... def ge3(x):
        ...     return x>=7
        >>> ge3.to_list()
        [8, 10]
    """

    def __init__(self, iterable: Iterable) -> None:
        """Initializes a new DataCollection instance.

        Args:
            iterable (Iterable): The iterable data that is stored in the DataCollection.
        """
        super().__init__()
        self._iterable = iterable

    def __iter__(self):
        """Generate an iterator of the DataCollection.

        Returns:
            iter : iterator for the data.
        """
        if hasattr(self._iterable, 'iterrows'):
            return (x[1] for x in self._iterable.iterrows())
        return iter(self._iterable)

    def __getattr__(self, name):
        """Unknown method dispatcher.

        When a unknown method is invoked on a `DataCollection` object, the function call
        will be dispatched to a method resolver. By registering function to the
        resolver, you are able to extend `DataCollection`'s API at runtime without
        modifying its code.

        Examples:
            >>> from towhee import register
            >>> dc = DataCollection([1,2,3,4])
            >>> @register(name='test/add1')
            ... def add1(x):
            ...     return x+1
            >>> dc.test.add1().to_list()
            [2, 3, 4, 5]

        Args:
            name (str): The unkown attribute.

        Returns:
            DataCollection: Returns a new DataCollection for the output of attribute
                call.
        """
        if name.startswith('_'):
            return super().__getattribute__(name)

        @dynamic_dispatch
        def wrapper(*arg, **kws):
            with param_scope() as hp:
                # pylint: disable=protected-access
                path = hp._name
                index = hp._index
            if self.get_backend() == 'ray':
                return self.ray_resolve({}, path, index, *arg, **kws)
            if self._jit is not None:
                op = self.jit_resolve(path, index, *arg, **kws)
            else:
                op = self.resolve(path, index, *arg, **kws)
            return self.map(op)

        return getattr(wrapper, name)

    def __getitem__(self, index):
        """Index based access of element in DataCollection.

        Access the element at the given index, similar to accessing `list[at_index]`.
        Does not work with streamed DataCollections.

        Examples:
            >>> dc = DataCollection([0, 1, 2, 3, 4])
            >>> dc[2]
            2

            >>> dc.stream()[1] # doctest: +NORMALIZE_WHITESPACE
            Traceback (most recent call last):
            TypeError: indexing is only supported for DataCollection created from list
                or pandas DataFrame.

        Args:
            index (int): The index location of the element being accessed.

        Raises:
            TypeError: If function called on streamed DataCollection

        Returns:
            any: The object at index.
        """
        if not hasattr(self._iterable, '__getitem__'):
            raise TypeError(
                'indexing is only supported for '
                'DataCollection created from list or pandas DataFrame.')
        if isinstance(index, int):
            return self._iterable[index]
        return DataCollection(self._iterable[index])

    def __setitem__(self, index, value):
        """Index based setting of element in DataCollection.

        Assign the value of the element at the given index, similar to
        `list[at_index]=val`. Does not work with streamed DataCollections.

        Examples:
            >>> dc = DataCollection([0, 1, 2, 3, 4])
            >>> dc[2] = 3
            >>> dc.to_list()
            [0, 1, 3, 3, 4]

            >>> dc.stream()[1] # doctest: +NORMALIZE_WHITESPACE
            Traceback (most recent call last):
            TypeError: indexing is only supported for DataCollection created from list
                or pandas DataFrame.

        Args:
            index (int): The index location of the element being set.
            val (any): The value to be set.

        Raises:
            TypeError: If function called on streamed DataCollection
        """
        if not hasattr(self._iterable, '__setitem__'):
            raise TypeError(
                'indexing is only supported for '
                'DataCollection created from list or pandas DataFrame.')
        self._iterable[index] = value

    @register_dag
    def __add__(self, other):
        """Concat two DataCollections.

        Args:
            other (DataCollection): The DataCollection being appended to the calling
                DataFrame.

        Examples:
            >>> (DataCollection.range(5) + DataCollection.range(5)).to_list()
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

            >>> dc0 = DataCollection.range(5)
            >>> dc1 = DataCollection.range(5)
            >>> dc2 = DataCollection.range(5)
            >>> (dc0 + dc1 + dc2)
            [0, 1, 2, 3, 4, 0, ...]

        Returns:
            DataCollection: A new DataCollection of the concated DataCollections.
        """
        self.parent_ids.append(other.id)
        other.notify_consumed(self.id)

        def inner():
            for x in self:
                yield x
            for x in other:
                yield x

        return self._factory(inner())

    def __repr__(self) -> str:
        """String representation of the DataCollection

        Examples:
            >>> DataCollection([1, 2, 3]).unstream()
            [1, 2, 3]

            >>> DataCollection([1, 2, 3]).stream() #doctest: +ELLIPSIS
            <list_iterator object at...>

        Returns:
            str: String repersentation of the DataCollection.
        """
        if isinstance(self._iterable, list):
            return reprlib.repr(self._iterable)
        if hasattr(self._iterable, '__repr__'):
            return repr(self._iterable)
        return super().__repr__()

    # Generation Related Function
    def _factory(self, iterable, parent_stream=True):
        """Factory method for Creating new DataCollections.

        This factory method has been wrapped into a `param_scope()` which contains the
        parent DataCollection's information.

        Args:
            iterable (Iterable): The data being encapsulated by the DataCollection
            parent_stream (bool, optional): Whether to use the same format of parent
                DataCollection (streamed or unstreamed). Defaults to True.

        Returns:
            DataCollection: The newly created DataCollection.
        """
        if parent_stream is True:
            if self.is_stream:
                if not isinstance(iterable, Iterator):
                    iterable = iter(iterable)
            else:
                if isinstance(iterable, Iterator):
                    iterable = list(iterable)

        with param_scope() as hp:
            hp().data_collection.parent = self
            return DataCollection(iterable)

    @staticmethod
    @register_dag
    def range(*arg, **kws):
        """Generate DataCollection with range of values.

        Generate DataCollection with a range of numbers as the data. Functions in same
        way as Python `range()` function.

        Examples:
            >>> DataCollection.range(5).to_list()
            [0, 1, 2, 3, 4]

        Returns:
            DataCollection: Returns a new DataCollection.
        """
        return DataCollection(range(*arg, **kws))

    def to_list(self):
        """Convert DataCollection to list.

        Examples:
            >>> DataCollection.range(5).to_list()
            [0, 1, 2, 3, 4]

        Returns:
            list: List of values stored in DataCollection.
        """
        return self._iterable if isinstance(self._iterable, list) else list(self._iterable)

    @register_dag
    def map(self, *arg):
        """Apply a function across all values in a DataCollection.

        Can apply multiple functions to the DataCollection. If multiple functions
        supplied, the same amount of new DataCollections will be returend.

        Examples:
            1. Single Functions::
            >>> dc = DataCollection([1,2,3,4])
            >>> dc.map(lambda x: x+1).map(lambda x: x*2).to_list()
            [4, 6, 8, 10]

            2. Multiple Functions::
            >>> dc = DataCollection([1,2,3,4])
            >>> a, b = dc.map(lambda x: x+1, lambda x: x*2)
            >>> (a.to_list(), b.to_list())
            ([2, 3, 4, 5], [2, 4, 6, 8])

        Args:
            *arg (Callable): One or multiple functions to apply to the DataCollection.

        Returns:
            DataCollection: New DataCollection containing computation results.
        """
        # mmap
        if len(arg) > 1:
            return self.mmap(list(arg))
        unary_op = arg[0]

        # smap map for stateful operator
        if hasattr(unary_op, 'is_stateful') and unary_op.is_stateful:
            return self.smap(unary_op)

        # pmap
        if self.get_executor() is not None:
            return self.pmap(unary_op)

        if hasattr(self._iterable, 'map'):
            return self._factory(self._iterable.map(unary_op))

        if hasattr(self._iterable, 'apply') and hasattr(unary_op, '__dataframe_apply__'):
            return self._factory(unary_op.__dataframe_apply__(self._iterable))

        # map
        def inner(x):
            if isinstance(x, Option):
                return x.map(unary_op)
            else:
                return unary_op(x)

        result = map(inner, self._iterable)
        return self._factory(result)

    @register_dag
    def filter(self, unary_op: Callable, drop_empty=False) -> 'DataCollection':
        """Filter the DataCollection data based on function.

        Filters the DataCollection based on the function provided. If data is stored
        as an Option (see towhee.functional.option.py), drop empty will decide whether
        to remove the element or set it to empty.

        Args:
            unary_op (Callable): Function that dictates filtering.
            drop_empty (bool, optional): Whether to drop empty fields. Defaults to False.

        Returns:
            DataCollection: Resulting DataCollection after filter.
        """
        def inner(x):
            if isinstance(x, Option):
                if isinstance(x, Some):
                    return unary_op(x.get())
                return not drop_empty
            return unary_op(x)

        if hasattr(self._iterable, 'filter'):
            return self._factory(self._iterable.filter(unary_op))

        if hasattr(self._iterable, 'apply') and hasattr(unary_op, '__dataframe_filter__'):
            return DataCollection(unary_op.__dataframe_apply__(self._iterable))

        return self._factory(filter(inner, self._iterable))

    def run(self):
        """Consume the iterables if in stream mode.
        """
        for _ in self._iterable:
            pass

    def to_df(self):
        """Turn a DataCollection into a DataFrame.

        Examples:
            >>> from towhee import DataCollection, Entity
            >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
            >>> dc = DataCollection(e)
            >>> type(dc)
            <class 'towhee.functional.data_collection.DataCollection'>
            >>> type(dc.to_df())
            <class 'towhee.functional.data_collection.DataFrame'>

        Returns:
            DataFrame: Resulting converted DataFrame.
        """
        return DataFrame(self._iterable)


class DataFrame(DataCollection, DataFrameMixin, ColumnMixin):
    """Entity based DataCollection.

    Examples:
        >>> from towhee import Entity
        >>> DataFrame([Entity(id=a) for a in [1,2,3]])
        [<Entity dict_keys(['id'])>, <Entity dict_keys(['id'])>, <Entity dict_keys(['id'])>]
    """

    def __init__(self, iterable: Iterable = None, **kws) -> None:
        """Initializes a new DataFrame instance.

        Args:
            iterable (Iterable, optional): The data to be encapsualted by the DataFrame.
                Defaults to None.
        """
        if iterable is not None:
            super().__init__(iterable)
            self._mode = self.ModeFlag.ROWBASEDFLAG
        else:
            super().__init__(DataFrame.from_arrow_talbe(**kws))
            self._mode = self.ModeFlag.COLBASEDFLAG


    def _factory(self, iterable, parent_stream=True, mode=None):
        """Factory method for Creating new DataFrames.

        This factory method has been wrapped into a `param_scope()` which contains the
        parent DataFrames's information.

        Args:
            iterable (Iterable): The data being encapsulated by the DataFrame
            parent_stream (bool, optional): Whether to use the same format of parent
                DataFrame (streamed or unstreamed). Defaults to True.
            mode (ModeFlag): The storage mode of the Dataframe.

        Returns:
            DataFrame: The newly created DataFrame.
        """

        # pylint: disable=protected-access
        if parent_stream is True:
            if self.is_stream:
                if not isinstance(iterable, Iterator):
                    iterable = iter(iterable)
            else:
                if isinstance(iterable, Iterator):
                    iterable = list(iterable)

        with param_scope() as hp:
            hp().data_collection.parent = self
            df = DataFrame(iterable)
            df._mode = self._mode if mode is None else mode
            return df

    def to_dc(self):
        """Turn a DataFrame into a DataCollection.

        Examples:
            >>> from towhee import DataFrame, Entity
            >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
            >>> df = DataFrame(e)
            >>> type(df)
            <class 'towhee.functional.data_collection.DataFrame'>
            >>> type(df.to_dc())
            <class 'towhee.functional.data_collection.DataCollection'>

        Returns:
            DataCollection: Resulting DataCollection from DataFrame
        """
        return DataCollection(self._iterable)

    @property
    def mode(self):
        """Storage mode of the DataFrame.

        Return the storage mode of the DataFrame.

        Examples:
            >>> from towhee import Entity, DataFrame
            >>> e = [Entity(a=a, b=b) for a,b in zip(range(5), range(5))]
            >>> df = DataFrame(e)
            >>> df.mode
            <ModeFlag.ROWBASEDFLAG: 1>
            >>> df = df.to_column()
            >>> df.mode
            <ModeFlag.COLBASEDFLAG: 2>

        Returns:
            ModeFlag: The storage format of the Dataframe.
        """
        return self._mode

    def __iter__(self):
        """Generate an iterator of the DataFrame.

        Examples:
            1. Row Based::
            >>> from towhee import Entity, DataFrame
            >>> e = [Entity(a=a, b=b) for a,b in zip(range(3), range(3))]
            >>> df = DataFrame(e)
            >>> df.to_list()[0]
            <Entity dict_keys(['a', 'b'])>

            2. Column Based::
            >>> df = df.to_column()
            >>> df.to_list()[0]
            <EntityView dict_keys(['a', 'b'])>

            2. Chunk Bassed::
            >>> df = DataFrame(e)
            >>> df = df.set_chunksize(2)
            >>> df.to_list()[0]
            <EntityView dict_keys(['a', 'b'])>

        Returns:
            iterator: The iterator for the DataFrame.
        """
        if hasattr(self._iterable, 'iterrows'):
            return (x[1] for x in self._iterable.iterrows())
        if self._mode == self.ModeFlag.ROWBASEDFLAG:
            return iter(self._iterable)
        if self._mode == self.ModeFlag.COLBASEDFLAG:
            return (EntityView(i, self._iterable) for i in range(len((self._iterable))))
        if self._mode == self.ModeFlag.CHUNKBASEDFLAG:
            return (ev for wtable in self._iterable.chunks() for ev in wtable)

    def map(self, *arg):
        """Apply a function across all values in a DataFrame.

        Args:
            *arg (Callable): One function to apply to the DataFrame.

        Returns:
            DataFrame: New DataFrame containing computation results.
        """
        if hasattr(arg[0], '__check_init__'):
            arg[0].__check_init__()
        if self._mode == self.ModeFlag.COLBASEDFLAG or self._mode == self.ModeFlag.CHUNKBASEDFLAG:
            return self.cmap(arg[0])
        else:
            return super().map(*arg)

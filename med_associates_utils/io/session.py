from collections import Counter
from typing import Any, Callable, List, Literal, Union

import numpy as np
import pandas as pd


FieldList = Union[Literal["all"], list[str]]


class Session(object):
    """Holds data and metadata for a single session."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.scalars: dict[str, float] = {}
        self.arrays: dict[str, np.ndarray] = {}

    def describe(self, as_str: bool = False) -> Union[str, None]:
        """Describe this session

        describes the metadata, scalars, and arrays contained in this session.

        Parameters:
        as_str: if True, return description as a string, otherwise print the description and return None

        Returns:
        `None` if `as_str` is `False`; if `as_str` is `True`, returns the description as a `str`
        """
        buffer = ""

        buffer += "Metadata:\n"
        if len(self.metadata) > 0:
            for k, v in self.metadata.items():
                buffer += f"    {k}: {v}\n"
        else:
            buffer += '    < No Metadata Available >\n'
        buffer += "\n"

        buffer += "Scalars:\n"
        if len(self.scalars) > 0:
            for k, v in self.scalars.items():
                buffer += f'    "{k}": {v}\n'
        else:
            buffer += '    < No Scalars Available >\n'
        buffer += "\n"

        buffer += "Arrays:\n"
        if len(self.arrays) > 0:
            for k, v in self.arrays.items():
                buffer += f'    "{k}" with shape {v.shape}:\n    {np.array2string(v, prefix="    ")}\n\n'
        else:
            buffer += '    < No Arrays Available >\n'
        buffer += "\n"

        if as_str:
            return buffer
        else:
            print(buffer)
            return None

    def rename_array(self, old_name: str, new_name: str):
        """Rename a data array, from `old_name` to `new_name`.

        Raises an error if the new array name already exists.

        Parameters:
        old_name: the current name for the array
        new_name: the new name for the array
        """
        if new_name in self.arrays:
            raise KeyError(f"Key `{new_name}` already exists in data!")

        self.arrays[new_name] = self.arrays[old_name]
        self.arrays.pop(old_name)

    def rename_scalar(self, old_name: str, new_name: str):
        """Rename a scalar, from `old_name` to `new_name`.

        Raises an error if the new scalar name already exists.

        Parameters:
        old_name: the current name for the scalar
        new_name: the new name for the scalar
        """
        if new_name in self.scalars:
            raise KeyError(f"Key `{new_name}` already exists in data!")

        self.scalars[new_name] = self.scalars[old_name]
        self.scalars.pop(old_name)

    def to_dataframe(self, include_arrays: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with array data and metadata.

        Parameters:
        include_arrays: list of array names to include in the dataframe. Special str "all" is also accepted.
        include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
        DataFrame with data from this session
        """
        # determine metadata fileds to include
        if include_meta == "all":
            meta = self.metadata
        else:
            meta = {k: v for k, v in self.metadata.items() if k in include_meta}

        # determine arrays to include
        if include_arrays == "all":
            array_names = list(self.arrays.keys())
        else:
            array_names = [k for k in self.arrays.keys() if k in include_arrays]

        # iterate arrays and include any the user requested
        # also add in any requested metadata
        events = []
        for k, v in self.arrays.items():
            if k in array_names:
                for value in v:
                    events.append({**meta, "event": k, "time": value})

        df = pd.DataFrame(events)

        # sort the dataframe by time, but check that we have values, otherwise will raise keyerror
        if len(df.index) > 0:
            df = df.sort_values("time")

        return df

    def scalar_dataframe(self, include_scalars: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with scalar data and metadata.

        Parameters:
        include_scalars: list of scalar names to include in the dataframe. Special str "all" is also accepted.
        include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
        DataFrame with data from this session
        """
        # determine metadata fileds to include
        if include_meta == "all":
            meta = self.metadata
        else:
            meta = {k: v for k, v in self.metadata.items() if k in include_meta}

        # determine scalars to include
        if include_scalars == "all":
            scalar_names = list(self.scalars.keys())
        else:
            scalar_names = [k for k in self.scalars.keys() if k in include_scalars]

        scalars = []
        for sn in scalar_names:
            scalars.append({**meta, "scalar_name": sn, "scalar_value": self.scalars[sn]})

        return pd.DataFrame(scalars)


class SessionCollection(list[Session]):
    """Collection of session data"""

    @property
    def metadata(self) -> pd.DataFrame:
        """Get a dataframe containing metadata across all sessions in this collection."""
        return pd.DataFrame([item.metadata for item in self])

    @property
    def metadata_keys(self) -> List[str]:
        """Get a list of the keys present in metadata across all sessions in this collection"""
        return list(set([key for item in self for key in item.metadata.keys()]))

    def add_metadata(self, key: str, value: Any) -> None:
        """Set a metadata field on each session in this collection

        Parameters:
        key: name of the metadata field
        value: value for the metadata field
        """
        for item in self:
            item.metadata[key] = value

    def update_metadata(self, meta: dict[str, Any]) -> None:
        """Set multiple metadata fields on each session in this collection

        Parameters:
        meta: metadata information to set on each session
        """
        for item in self:
            item.metadata.update(meta)

    def rename_array(self, old_name: str, new_name: str) -> None:
        """Rename an array on each session in this collection

        Parameters:
        old_name: current name of the array
        new_name: the new name for the array
        """
        for item in self:
            item.rename_array(old_name, new_name)

    def rename_scalar(self, old_name: str, new_name: str) -> None:
        """Rename an scalar on each session in this collection

        Parameters:
        old_name: current name of the scalar
        new_name: the new name for the scalar
        """
        for item in self:
            item.rename_scalar(old_name, new_name)

    def filter(self, predicate: Callable[[Session], bool]) -> "SessionCollection":
        """Filter the items in this collection, returning a new `SessionCollection` containing sessions which pass `predicate`.

        Parameters:
        predicate: a callable accepting a single session and returning bool.

        Returns:
        a new `SessionCollection` containing only itemss which pass `predicate`.
        """
        return type(self)(item for item in self if predicate(item))

    def map(self, action: Callable[[Session], Session]) -> "SessionCollection":
        """Apply a function to each session in this collection, returning a new collection with the results

        Parameters:
        action: callable accepting a single session and returning a new session

        Returns:
        a new `SessionCollection` containing the results of `action`
        """
        return type(self)(action(item) for item in self)

    def apply(self, func: Callable[[Session], None]) -> None:
        """Apply a function to each session in this collection

        Parameters:
        func: callable accepting a single session and returning None
        """
        for item in self:
            func(item)

    def get_array(self, name: str) -> list[np.ndarray]:
        """Get data across sessions in this collection for the array named `name`

        Parameters:
        name: Name of the arrays to collect

        Returns
        List of numpy arrays, each corresponding to a single session
        """
        return [item.arrays[name] for item in self]

    def describe(self, as_str: bool = False) -> Union[str, None]:
        """Describe this collection of sessions

        Parameters:
        as_str: if True, return description as a string, otherwise print the description and return None

        Returns:
        `None` if `as_str` is `False`; if `as_str` is `True`, returns the description as a `str`
        """
        buffer = ""

        buffer += f"Number of sessions: {len(self)}\n\n"

        arrays = Counter([item for session in self for item in session.arrays.keys()])
        buffer += "Arrays present in data with counts:\n"
        for k, v in arrays.items():
            buffer += f'({v}) "{k}"\n'
        buffer += "\n"

        scalars = Counter([item for session in self for item in session.scalars.keys()])
        buffer += "Scalars present in data with counts:\n"
        for k, v in scalars.items():
            buffer += f'({v}) "{k}"\n'
        buffer += "\n"

        if as_str:
            return buffer
        else:
            print(buffer)
            return None

    def to_dataframe(self, include_arrays: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with array data and metadata across all the sessions in this collection.

        Parameters:
        include_arrays: list of array names to include in the dataframe. Special str "all" is also accepted.
        include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
        DataFrame with data from across this collection
        """
        dfs = [session.to_dataframe(include_arrays=include_arrays, include_meta=include_meta) for session in self]
        return pd.concat(dfs).sort_values("time").reset_index(drop=True)

    def scalar_dataframe(self, include_scalars: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with array data and metadata across all the sessions in this collection.

        Parameters:
        include_scalars: list of scalar names to include in the dataframe. Special str "all" is also accepted.
        include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
        DataFrame with data from across this collection
        """
        dfs = [session.scalar_dataframe(include_scalars=include_scalars, include_meta=include_meta) for session in self]
        return pd.concat(dfs).reset_index(drop=True)

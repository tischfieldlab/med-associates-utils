import datetime
import glob
import os
import re
from collections import Counter
from typing import Any, Callable, List, Literal, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

FieldList = Union[Literal['all'], list[str]]

class MPCSession(object):
    '''Holds data and metadata for a single session.
    '''

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.scalars: dict[str, float] = {}
        self.arrays: dict[str, np.ndarray] = {}


    def rename_array(self, old_name: str, new_name: str):
        ''' Rename a data array, from `old_name` to `new_name`.

        Raises an error if the new array name already exists.

        Parameters:
        old_name: the current name for the array
        new_name: the new name for the array
        '''
        if new_name in self.arrays:
            raise KeyError(f'Key `{new_name}` already exists in data!')

        self.arrays[new_name] = self.arrays[old_name]
        self.arrays.pop(old_name)


    def rename_scalar(self, old_name: str, new_name: str):
        ''' Rename a scalar, from `old_name` to `new_name`.

        Raises an error if the new scalar name already exists.

        Parameters:
        old_name: the current name for the scalar
        new_name: the new name for the scalar
        '''
        if new_name in self.scalars:
            raise KeyError(f'Key `{new_name}` already exists in data!')

        self.scalars[new_name] = self.scalars[old_name]
        self.scalars.pop(old_name)


    def to_dataframe(self, include_arrays: FieldList = 'all', include_meta: FieldList = 'all') -> pd.DataFrame:
        '''Produce a dataframe with array data and metadata.

        Parameters:
        include_arrays: list of array names to include in the dataframe. Special str "all" is also accepted.
        include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
        DataFrame with data from this session
        '''
        # determine metadata fileds to include
        if include_meta == 'all':
            meta = self.metadata
        else:
            meta = {k: v for k, v in self.metadata.items() if k in include_meta}

        # determine arrays to include
        if include_arrays == 'all':
            array_names = list(self.arrays.keys())
        else:
            array_names = [k for k in self.arrays.keys() if k in include_arrays]

        # iterate arrays and include any the user requested
        # also add in any requested metadata
        events = []
        for k, v in self.arrays.items():
            if k in array_names:
                for value in v:
                    events.append({
                        **meta,
                        'event': k,
                        'time': value
                    })

        # finally sort the dataframe by time and return it
        return pd.DataFrame(events).sort_values('time')


class SessionCollection(list[MPCSession]):
    '''Collection of session data'''

    @property
    def metadata(self) -> pd.DataFrame:
        '''Get a dataframe containing metadata across all sessions in this collection.
        '''
        return pd.DataFrame([item.metadata for item in self])

    def add_metadata(self, key: str, value: Any) -> None:
        '''Set a metadata field on each session in this collection

        Parameters:
        key: name of the metadata field
        value: value for the metadata field
        '''
        for item in self:
            item.metadata[key] = value

    def update_metadata(self, meta: dict[str, Any]) -> None:
        '''Set multiple metadata fields on each session in this collection

        Parameters:
        meta: metadata information to set on each session
        '''
        for item in self:
            item.metadata.update(meta)

    def rename_array(self, old_name: str, new_name: str) -> None:
        '''Rename an array on each session in this collection

        Parameters:
        old_name: current name of the array
        new_name: the new name for the array
        '''
        for item in self:
            item.rename_array(old_name, new_name)

    def rename_scalar(self, old_name: str, new_name: str) -> None:
        '''Rename an scalar on each session in this collection

        Parameters:
        old_name: current name of the scalar
        new_name: the new name for the scalar
        '''
        for item in self:
            item.rename_scalar(old_name, new_name)

    def filter(self, predicate: Callable[[MPCSession], bool]) -> 'SessionCollection':
        '''Filter the items in this collection, returning a new `SessionCollection` containing sessions which pass `predicate`.

        Parameters:
        predicate: a callable accepting a single session and returning bool.

        Returns:
        a new `SessionCollection` containing only itemss which pass `predicate`.
        '''
        return type(self)(item for item in self if predicate(item))

    def map(self, action: Callable[[MPCSession], MPCSession]) -> 'SessionCollection':
        '''Apply a function to each session in this collection, returning a new collection with the results

        Parameters:
        action: callable accepting a single session and returning a new session

        Returns:
        a new `SessionCollection` containing the results of `action`
        '''
        return type(self)(action(item) for item in self)

    def apply(self, func: Callable[[MPCSession], None]) -> None:
        '''Apply a function to each session in this collection

        Parameters:
        func: callable accepting a single session and returning None
        '''
        for item in self:
            func(item)

    def get_array(self, name: str) -> list[np.ndarray]:
        '''Get data across sessions in this collection for the array named `name`

        Parameters:
        name: Name of the arrays to collect

        Returns
        List of numpy arrays, each corresponding to a single session
        '''
        return [item.arrays[name] for item in self]

    def describe(self, as_str: bool = False) -> Union[str, None]:
        '''Describe this collection

        Parameters:
        as_str: if True, return description as a string, otherwise print the description and return None

        Returns:
        `None` if `as_str` is `False`; if `as_str` is `True`, returns the description as a `str`
        '''
        buffer = ""

        buffer += f'Number of sessions: {len(self)}\n\n'

        arrays = Counter([item for session in self for item in session.arrays.keys()])
        buffer += 'Arrays present in data with counts:\n'
        for k, v in arrays.items():
            buffer += f'({v}) "{k}"\n'
        buffer += '\n'

        scalars = Counter([item for session in self for item in session.scalars.keys()])
        buffer += 'Scalars present in data with counts:\n'
        for k, v in scalars.items():
            buffer += f'({v}) "{k}"\n'
        buffer += '\n'

        if as_str:
            return buffer
        else:
            print(buffer)
            return None

    def to_dataframe(self, include_arrays: FieldList = 'all', include_meta: FieldList = 'all') -> pd.DataFrame:
        '''Produce a dataframe with array data and metadata across all the sessions in this collection.

        Parameters:
        include_arrays: list of array names to include in the dataframe. Special str "all" is also accepted.
        include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
        DataFrame with data from across this collection
        '''
        dfs = [session.to_dataframe(include_arrays=include_arrays, include_meta=include_meta) for session in self]
        return pd.concat(dfs).sort_values('time').reset_index(drop=True)



_rx_dict = {
    'StartDate': re.compile(r'^Start Date: (?P<StartDate>.*)\r\n'),
    'EndDate': re.compile(r'^End Date: (?P<EndDate>.*)\r\n'),
    'StartTime': re.compile(r'^Start Time: (?P<StartTime>.*)\r\n'),
    'EndTime': re.compile(r'^End Time: (?P<EndTime>.*)\r\n'),
    'Subject': re.compile(r'^Subject: (?P<Subject>.*)\r\n'),
    'Experiment': re.compile(r'^Experiment: (?P<Experiment>.*)\r\n'),
    'Group': re.compile(r'^Group: (?P<Group>.*)\r\n'),
    'Box': re.compile(r'^Box: (?P<Box>.*)\r\n'),
    'MSN': re.compile(r'^MSN: (?P<MSN>.*)\r\n'),
    'SCALAR': re.compile(r'(?P<name>[A-Z]{1}): *(?P<value>\d+\.\d*)\r\n'),
    'ARRAY': re.compile(r'(?P<name>[A-Z]{1}):\r\n'),
    'ARRAYidx': re.compile(r'^ *(?P<index>[0-9]+):(?P<list>.*)\r\n'),
    'STARTOFDATA': re.compile(r'\r\r\n')
}

def _parse_line(line: str):
    '''Parse a single session data file line.

    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex
    '''
    for key, rx in _rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None


def parse_directory(path: str, pattern: str = '*.txt') -> SessionCollection:
    '''Parse a directory containing session data files from MedAssociates

    Parameters:
    path: path to directory containing data files
    pattern: glob pattern for selecting files in the directory

    Returns:
    SessionCollection with parsed files
    '''
    sessions = SessionCollection()
    for filepath in tqdm(glob.glob(os.path.join(path, pattern))):
        sessions.extend(parse_session(filepath))
    return sessions


def parse_session(filepath: str) -> SessionCollection:
    '''Parse a session data file from MedAssociates

    Adapted from https://github.com/matthewperkins/MPCdata, but fixes some issues.

    Parameters:
    filepath: Filepath for file_object to be parsed

    Returns:
    SessionCollection
    '''
    #print(filepath)
    data: MPCSession
    MPCDateStringRe = re.compile(r'\s*(?P<hour>[0-9]+):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})')
    # open the file and read through it line by line
    with open(filepath, 'r', newline = '\n') as file_object:
        # if the file has multiple boxes in it, return a list of MPC objects
        MPCDataList = SessionCollection()
        line = file_object.readline()
        while line:
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # start of data is '\r\r\n'
            if key=='STARTOFDATA':
                data = MPCSession()  # create a new data object
                MPCDataList.append(data)

            # extract start date
            if key == 'StartDate':
                data.metadata['StartDate'] = datetime.datetime.strptime(match.group(key), "%m/%d/%y").date()

            # extract end date
            if key == 'EndDate':
                data.metadata['EndDate'] = datetime.datetime.strptime(match.group(key), "%m/%d/%y").date()

            # extract start time
            if key == 'StartTime':
                (h,m,s) = [int(MPCDateStringRe.search(match.group(key)).group(g)) for g in ['hour', 'minute', 'second']]
                data.metadata['StartTime'] = datetime.time(h,m,s)
                # date should be already read
                data.metadata['StartDateTime'] = datetime.datetime.combine(data.metadata['StartDate'], data.metadata['StartTime'])

            # extract end time
            if key == 'EndTime':
                (h,m,s) = [int(MPCDateStringRe.search(match.group(key)).group(g)) for g in ['hour', 'minute', 'second']]
                data.metadata['EndTime'] = datetime.time(h,m,s)

            # extract Subject
            if key == 'Subject':
                data.metadata['Subject'] = match.group(key)

            # extract Experiment
            if key == 'Experiment':
                data.metadata['Experiment'] = match.group(key)

            # extract Group
            if key == 'Group':
                data.metadata['Group'] = match.group(key)

            # extract Box
            if key == 'Box':
                data.metadata['Box'] = int(match.group(key))

            # extract MSN
            if key == 'MSN':
                data.metadata['MSN'] = match.group(key)

            # extract scalars
            if key == 'SCALAR':
                data.scalars[match.group('name')] = float(match.group('value'))

            # identify an array
            if key == 'ARRAY':
                #print(f'This is the beginning of an Array:: "{line}"')
                # have now have to step through the array
                file_tell = file_object.tell()
                subline = file_object.readline()
                #print(f'This is the first line of the array:: "{subline}"')
                items = []
                while subline:
                    m = _rx_dict['ARRAYidx'].search(subline)
                    if (m):
                        items.extend([float(l) for l in m.group('list').split()])

                    else:
                        # have to rewind
                        #print(f'This is one line beyond the last line of the array:: "{subline}"')
                        file_object.seek(file_tell)
                        break
                    file_tell = file_object.tell()
                    subline = file_object.readline()
                #print(f'Setting "{match.group("name")}"={items}')
                data.arrays[match.group('name')] = np.array(items)
            line = file_object.readline()
    return MPCDataList

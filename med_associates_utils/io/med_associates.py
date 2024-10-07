import datetime
import glob
import os
import re

import numpy as np
from tqdm.auto import tqdm

from .session import Session, SessionCollection


_rx_dict = {
    "StartDate": re.compile(r"^Start Date: (?P<StartDate>.*)\r\n"),
    "EndDate": re.compile(r"^End Date: (?P<EndDate>.*)\r\n"),
    "StartTime": re.compile(r"^Start Time: (?P<StartTime>.*)\r\n"),
    "EndTime": re.compile(r"^End Time: (?P<EndTime>.*)\r\n"),
    "Subject": re.compile(r"^Subject: (?P<Subject>.*)\r\n"),
    "Experiment": re.compile(r"^Experiment: (?P<Experiment>.*)\r\n"),
    "Group": re.compile(r"^Group: (?P<Group>.*)\r\n"),
    "Box": re.compile(r"^Box: (?P<Box>.*)\r\n"),
    "MSN": re.compile(r"^MSN: (?P<MSN>.*)\r\n"),
    "SCALAR": re.compile(r"(?P<name>[A-Z]{1}): *(?P<value>\d+\.\d*)\r\n"),
    "ARRAY": re.compile(r"(?P<name>[A-Z]{1}):\r\n"),
    "ARRAYidx": re.compile(r"^ *(?P<index>[0-9]+):(?P<list>.*)\r\n"),
    "STARTOFDATA": re.compile(r"\r\r\n"),
}


def _parse_line(line: str):
    """Parse a single session data file line.

    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex
    """
    for key, rx in _rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None


def parse_ma_directory(path: str, pattern: str = "*.txt", quiet: bool = False) -> SessionCollection:
    """Parse a directory containing session data files from MedAssociates

    Parameters:
    path: path to directory containing data files
    pattern: glob pattern for selecting files in the directory
    quiet: if false, show TQDM progress bar, if True, do not show any progress bar

    Returns:
    SessionCollection with parsed files
    """
    sessions = SessionCollection()
    for filepath in tqdm(glob.glob(os.path.join(path, pattern)), disable=quiet, leave=True):
        sessions.extend(parse_ma_session(filepath))
    return sessions


def parse_ma_session(filepath: str) -> SessionCollection:
    """Parse a session data file from MedAssociates

    Adapted from https://github.com/matthewperkins/MPCdata, but fixes some issues.

    Parameters:
    filepath: Filepath for file_object to be parsed

    Returns:
    SessionCollection
    """
    # print(filepath)
    data: Session
    MPCDateStringRe = re.compile(r"\s*(?P<hour>[0-9]+):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})")
    # open the file and read through it line by line
    with open(filepath, "r", newline="\n") as file_object:
        # if the file has multiple boxes in it, return a list of MPC objects
        MPCDataList = SessionCollection()
        line = file_object.readline()
        while line:
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # start of data is '\r\r\n'
            if key == "STARTOFDATA":
                data = Session()  # create a new data object
                MPCDataList.append(data)

            # extract start date
            if key == "StartDate":
                data.metadata["StartDate"] = datetime.datetime.strptime(match.group(key), "%m/%d/%y").date()

            # extract end date
            if key == "EndDate":
                data.metadata["EndDate"] = datetime.datetime.strptime(match.group(key), "%m/%d/%y").date()

            # extract start time
            if key == "StartTime":
                date_match = MPCDateStringRe.search(match.group(key))
                if date_match is not None:
                    (hour, min, sec) = [int(date_match.group(g)) for g in ["hour", "minute", "second"]]
                    data.metadata["StartTime"] = datetime.time(hour, min, sec)
                    # date should be already read
                    data.metadata["StartDateTime"] = datetime.datetime.combine(data.metadata["StartDate"], data.metadata["StartTime"])

            # extract end time
            if key == "EndTime":
                date_match = MPCDateStringRe.search(match.group(key))
                if date_match is not None:
                    (hour, min, sec) = [int(date_match.group(g)) for g in ["hour", "minute", "second"]]
                    data.metadata["EndTime"] = datetime.time(hour, min, sec)
                    # date should be already read
                    data.metadata["EndDateTime"] = datetime.datetime.combine(data.metadata["EndDate"], data.metadata["EndTime"])

            # extract Subject
            if key == "Subject":
                data.metadata["Subject"] = match.group(key)

            # extract Experiment
            if key == "Experiment":
                data.metadata["Experiment"] = match.group(key)

            # extract Group
            if key == "Group":
                data.metadata["Group"] = match.group(key)

            # extract Box
            if key == "Box":
                data.metadata["Box"] = int(match.group(key))

            # extract MSN
            if key == "MSN":
                data.metadata["MSN"] = match.group(key)

            # extract scalars
            if key == "SCALAR":
                data.scalars[match.group("name")] = float(match.group("value"))

            # identify an array
            if key == "ARRAY":
                # print(f'This is the beginning of an Array:: "{line}"')
                # have now have to step through the array
                file_tell = file_object.tell()
                subline = file_object.readline()
                # print(f'This is the first line of the array:: "{subline}"')
                items = []
                while subline:
                    m = _rx_dict["ARRAYidx"].search(subline)
                    if m is not None:
                        items.extend([float(l) for l in m.group("list").split()])

                    else:
                        # have to rewind
                        # print(f'This is one line beyond the last line of the array:: "{subline}"')
                        file_object.seek(file_tell)
                        break
                    file_tell = file_object.tell()
                    subline = file_object.readline()
                # print(f'Setting "{match.group("name")}"={items}')
                data.arrays[match.group("name")] = np.array(items)
            line = file_object.readline()
    return MPCDataList

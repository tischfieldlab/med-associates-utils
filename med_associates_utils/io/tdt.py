import glob
import os

import tdt
from tqdm.auto import tqdm

from .session import SessionCollection, Session


def parse_tdt_directory(path: str, pattern: str = "*", quiet: bool = False) -> SessionCollection:
    """Parse a directory containing data tanks from TDT

    Parameters:
    path: path to directory containing data files
    pattern: glob pattern for selecting files in the directory

    Returns:
    SessionCollection with parsed files
    """
    sessions = SessionCollection()
    for filepath in tqdm(glob.glob(os.path.join(path, pattern)), disable=quiet, leave=True):
        sessions.extend(parse_tdt_session(filepath))
    return sessions


def parse_tdt_session(filepath: str, **kwargs) -> SessionCollection:
    """Parse a TDT tank

    Parameters:
    filepath: Filepath for tank to be parsed
    **kwargs: keyword arguments passed to tdt.read_block()

    Returns:
    SessionCollection
    """
    # Open the tank and read data
    # by default, we ask to only unpack "epocs" event types, which is much faster
    # but we allow the user to pass their own kwargs, which might overwrite this
    default_kwargs = {
        'evtype': ['epocs']
    }
    data = tdt.read_block(filepath, **{**default_kwargs, **kwargs})

    session = Session()
    # add info items from the tank to the session metadata
    session.metadata.update(data.info.items())

    # loop through the epocs keys and add each item as an array to the session
    # here we only pay attention to the `onset`, but this might not always be what you want!
    for k in data.epocs.keys():
        session.arrays[k] = data.epocs[k].onset

    # return a SessionCollection to be compatible with MedAssociates parsing (where multi-session files are possible)
    return SessionCollection([session])

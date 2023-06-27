from typing import Union
import pandas as pd



def calculate_cumulative_events(df: pd.DataFrame, event: Union[str, list[str]], groupby: Union[str, list[str]], time_key: str = 'time')-> pd.DataFrame:
    df = df.copy()  # make a copy
    df = df[df['event'] == event].sort_values(by=time_key)  # filter to only the requested event(s)
    df[f'cum_{event}'] = df.groupby(by=groupby).cumcount() + 1
    df = df.drop(columns='event')
    return df

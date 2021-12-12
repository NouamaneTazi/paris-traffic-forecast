import os
import pandas as pd
import os
# Loads the data from the specified file and returns a pandas dataframe
def load_data(path_to_data='data/AV-champs-elysees.csv'):
    path_to_data = os.path.abspath(path_to_data)
    time_col = "Date et heure de comptage"
    df = pd.read_csv(path_to_data, sep=';').sort_values(time_col)
    # set datetime index
    df['Date et heure de comptage'] = pd.to_datetime(df['Date et heure de comptage'], utc=True).dt.tz_localize(None)
    df[time_col] = pd.DatetimeIndex(df[time_col])
    df = df.set_index(time_col)
    return df
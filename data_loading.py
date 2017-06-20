import json
import pandas as pd


def get_df(file_name='data/realTweets.csv'):
    df = pd.read_csv(file_name)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index("timestamp", inplace=True)
    return df


def get_labels_df(df, file_key='realTweets'):
    labels = json.load(open('data/combined_multivariate_labels.json', 'r'))
    labels_df = pd.DataFrame(columns=df.columns)
    for value in labels[file_key]:
        for timestamps in value.values():
            for timestamp in timestamps:
                labels_df.loc[timestamp, value] = 1
    labels_df.sort_index(inplace=True)
    labels_df.fillna(0, inplace=True)
    return labels_df

import pandas as pd


class MultiAnomalyDetector(object):
    def __init__(self):
        self.time_series = None

    def get_window(self, n=10):
        """Return a window of the time series including the n most recent elements
        :param n: window length
        :return: pd.DataFrame of length n
        """
        return self.time_series.iloc[-n:]

    def handle_row(self, data_row):
        """
        Return a Series of anomaly scores with the same size as the given row
        :param data_row: pd.Series
        :return: pd.Series with anomaly scores
        """
        scores = {"timestamp": data_row._name}
        if self.time_series is None:
            self.time_series = pd.DataFrame(columns=data_row.keys())
        self.time_series = self.time_series.append(data_row)

        scores = self.score_row(scores, data_row)
        return pd.DataFrame.from_records([scores], index=['timestamp'])

    def score_row(self, scores, data_row):
        """
        Compute anomaly scores
        :param scores: dict with timestamp
        :param data_row: current Series of measurements
        :return: dict with timestamp and anomaly scores per value key
        """
        return NotImplementedError

def test():
    print("j")
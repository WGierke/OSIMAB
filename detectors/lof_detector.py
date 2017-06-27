#!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import division
import sys

sys.path.append('detectors')  # TODO: fix this
from base import MultiAnomalyDetector
from sklearn.neighbors.lof import LocalOutlierFactor


class LofDetector(MultiAnomalyDetector):
    def __init__(self, neighbors=50):
        super(LofDetector, self).__init__()
        self.neighbors = neighbors


    def score_row(self, scores, data_row):
        window = self.get_window(n=self.neighbors)
        if len(window) < self.neighbors:
            for value_col in data_row.keys():
                scores[value_col] = 0
            return scores
        clf = LocalOutlierFactor(n_neighbors=self.neighbors)
        y = clf.fit_predict(window)[-1]  # inliner is 1, outlier is -1
        for value_col in data_row.keys():
            scores[value_col] = -y
        return scores

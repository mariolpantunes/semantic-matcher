#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


def jaccard_string(a,b):
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union


class SemanticMathcer():
    def __init__(self):
        self.services = {}

    def add(self, service):
        key = service['id']
        if key not in self.services:
            self.services[key] = service['description']
    
    def match(self, query):
        scoreJaccardString = []
        for s,d in self.services.items():
            scoreJaccardString.append(jaccard_string(query, d))
        print(scoreJaccardString)
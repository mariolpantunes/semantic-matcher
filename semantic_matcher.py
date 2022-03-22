#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
from rapidfuzz.distance import Levenshtein


def jaccard_string(a,b):
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union


def jaccard_levenshtein(a, b, t:int):
    scores = []
    for i in range(len(a)):
        for j in range(len(b)):
            scores.append((i,j,Levenshtein.distance(a[i], b[j])))
    scores.sort(key=lambda x:x[2])
    used_a, used_b = [], []
    rv = 0
    for i,j,s in scores:
        if s <= t and i not in used_a and j not in used_b:
            rv += 1
            used_a.append(i)
            used_b.append(j)
    rv = rv / (len(a) + len(b) - rv)
    return rv


def cosine(a, b, distance, t, reverse=False):
    scores = []
    for i in range(len(a)):
        for j in range(len(b)):
            scores.append((i,j,distance(a[i], b[j])))
    scores.sort(key=lambda x:x[2], reverse=reverse)
    used_a, used_b = [], []
    va, vb = [], []
    for i,j,s in scores:
        if s <= t and i not in used_a and j not in used_b:
            va.append(1)
            vb.append(1)
            used_a.append(i)
            used_b.append(j)
    
    for i in range(len(a)-len(used_a)):
        va.append(1)
        vb.append(0)
    
    for i in range(len(b)-len(used_b)):
        va.append(0)
        vb.append(1)
    
    return np.dot(va, vb)/(np.linalg.norm(va)*np.linalg.norm(vb))

class SemanticMathcer():
    def __init__(self, jt:float=0.3, lt:int=2, ct:float=0.5):
        self.services = {}
        self.jt = jt
        self.lt = lt
        self.ct = ct

    def add(self, service):
        key = service['id']
        if key not in self.services:
            self.services[key] = service['description']
    
    def match(self, query):
        # Jaccard
        ## String
        scoreJaccardString = []
        for s,d in self.services.items():
            scoreJaccardString.append((s, jaccard_string(query, d)))
        scoreJaccardString = [(i,s) for (i,s) in scoreJaccardString if s >= self.jt]
        scoreJaccardString.sort(key=lambda x:x[1], reverse=True)
        
        ## Levenshtein
        scoreJaccardLevenshtein = []
        for s,d in self.services.items():
            scoreJaccardLevenshtein.append((s, jaccard_levenshtein(query, d, self.lt)))
        scoreJaccardLevenshtein = [(i, s) for (i, s) in scoreJaccardLevenshtein if s >= self.jt]
        scoreJaccardLevenshtein.sort(key=lambda x:x[1], reverse=True)

        #Cosine
        ## String
        scoreCosineString = []
        for s,d in self.services.items():
            scoreCosineString.append((s, cosine(query, d, lambda x,y: Levenshtein.distance(x, y), 0)))
        print(f'scoreCosineString = {scoreCosineString}')
        scoreCosineString = [(i,s) for (i,s) in scoreCosineString if s >= self.ct]
        scoreCosineString.sort(key=lambda x:x[1], reverse=True)

        ## Levenshtein
        scoreCosineLevenshtein = []
        for s,d in self.services.items():
            scoreCosineLevenshtein.append((s, cosine(query, d, lambda x,y: Levenshtein.distance(x, y), self.lt)))
        scoreCosineLevenshtein = [(i,s) for (i,s) in scoreCosineLevenshtein if s >= self.ct]
        scoreCosineLevenshtein.sort(key=lambda x:x[1], reverse=True)

        return {'jaccard':{'string':scoreJaccardString,'levenshtein':scoreJaccardLevenshtein},
        'cosine':{'string': scoreCosineString, 'levenshtein':scoreCosineLevenshtein}}
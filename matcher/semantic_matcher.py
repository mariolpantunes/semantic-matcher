# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import semantic.dp as dp
import semantic.corpus as corpus


from matcher.metrics import cosine, jaccard
from joblib import Parallel, delayed
from rapidfuzz.distance import Levenshtein


def pre_compute_scores(a, b, distance, t=0, reverse=False):
    scores = []
    for i in range(len(a)):
        for j in range(len(b)):
            score = distance(a[i], b[j])
            if score > t and reverse:
                scores.append((i, j, score))
            if score <=t and not reverse:
                scores.append((i, j, score))
    return scores


def jaccard_query(a, b, scores, reverse=False):
    scores.sort(key=lambda x:x[2], reverse=reverse)
    used_a, used_b = [], []
    rv = 0.0
    for i, j, s in scores:
        if i not in used_a and j not in used_b:
            rv += 1.0
            used_a.append(i)
            used_b.append(j)
        
        if i not in used_a and j not in used_b:
            rv += 1.0
            used_a.append(i)
            used_b.append(j)
    
    rv = rv / (len(a) + len(b) - rv)
    return rv


def cosine_query(a, b, scores, reverse=False):
    scores.sort(key=lambda x:x[2], reverse=reverse)
    used_a, used_b = [], []
    va, vb = [], []
    for i, j, s in scores:
        
        if i not in used_a and j not in used_b:
            va.append(1.0)
            vb.append(1.0)
            used_a.append(i)
            used_b.append(j)
    
    for i in range(len(a)-len(used_a)):
        va.append(1.0)
        vb.append(0.0)
    
    for i in range(len(b)-len(used_b)):
        va.append(0.0)
        vb.append(1.0)
    
    return cosine(va, vb)


class SemanticMathcer():
    def __init__(self, key:str, path:str, jt:float=0.5, lt:int=2, ct:float=0.5, st:float=0.1, n:int=3, k:int=3):
        self.services = {}
        self.jt = jt
        self.lt = lt
        self.ct = ct
        self.st = st
        self.model = dp.DPWCModel(corpus=corpus.WebCorpus(key, path), n=n, kl=8, c=dp.Cutoff.pareto20, latent=True, k=k)

    def add(self, service):
        key = service['id']
        if key not in self.services:
            self.services[key] = service['description']
            self.model.fit(service['description'])
    
    def match(self, query):
        scoreJaccardString = []
        scoreJaccardLevenshtein = []
        scoreJaccardSemantic = []
        scoreCosineString = []
        scoreCosineLevenshtein = []
        scoreCosineSemantic = []

        # for all services within the db compute the overall scores
        for s,d in self.services.items():
            
            scores_string = pre_compute_scores(query, d, lambda x,y: Levenshtein.distance(x, y))
            scores_levenshtein = pre_compute_scores(query, d, lambda x,y: Levenshtein.distance(x, y), t=self.lt)
            scores_semantic = pre_compute_scores(query, d, lambda x,y: self.model.predict(x, y), t=self.st, reverse=True)

            scoreJaccardString.append((s, jaccard_query(query, d, scores_string)))
            scoreJaccardLevenshtein.append((s, jaccard_query(query, d, scores_levenshtein)))
            scoreJaccardSemantic.append((s, jaccard_query(query, d, scores_semantic, reverse=True)))
            
            scoreCosineString.append((s, cosine_query(query, d, scores_string)))
            scoreCosineLevenshtein.append((s, cosine_query(query, d, scores_levenshtein)))
            scoreCosineSemantic.append((s, cosine_query(query, d, scores_semantic, reverse=True)))
        
        # Jaccard
        ## String
        scoreJaccardString = [(i,s) for (i,s) in scoreJaccardString if s >= self.jt]
        scoreJaccardString.sort(key=lambda x:x[1], reverse=True)
        
        ## Levenshtein
        scoreJaccardLevenshtein = [(i, s) for (i, s) in scoreJaccardLevenshtein if s >= self.jt]
        scoreJaccardLevenshtein.sort(key=lambda x:x[1], reverse=True)

        ## Semantic
        scoreJaccardSemantic = [(i, s) for (i, s) in scoreJaccardSemantic if s >= self.jt]
        scoreJaccardSemantic.sort(key=lambda x:x[1], reverse=True)

        #Cosine
        ## String
        scoreCosineString = [(i,s) for (i,s) in scoreCosineString if s >= self.ct]
        scoreCosineString.sort(key=lambda x:x[1], reverse=True)

        ## Levenshtein
        scoreCosineLevenshtein = [(i,s) for (i,s) in scoreCosineLevenshtein if s >= self.ct]
        scoreCosineLevenshtein.sort(key=lambda x:x[1], reverse=True)

        ## Semantic
        scoreCosineSemantic = [(i, s) for (i, s) in scoreCosineSemantic if s >= self.jt]
        scoreCosineSemantic.sort(key=lambda x:x[1], reverse=True)

        return {'jaccard':{'string':scoreJaccardString,'levenshtein':scoreJaccardLevenshtein, 'semantic': scoreJaccardSemantic},
        'cosine':{'string': scoreCosineString, 'levenshtein':scoreCosineLevenshtein, 'semantic':scoreCosineSemantic}}
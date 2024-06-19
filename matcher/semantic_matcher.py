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


def compute_jaccard_scores(a, b, distance, t=0, reverse=False):
    scores = []
    for i in range(len(a)):
        for j in range(len(b)):
            score = distance(a[i], b[j])
            if score > t and reverse:
                scores.append((i, j, score))
            if score <=t and not reverse:
                scores.append((i, j, score))
    return scores


def compute_cosine_similarity(keywords, query, service_vector, similarity):
    query_vector = [0.0] * len(service_vector)
    for q in query:
        current_query_vector = []
        for k in keywords:
            current_query_vector.append(similarity(q, k))
        done = False
        while not done:
            max_value = max(current_query_vector)
            if max_value > 0:
                max_idx = current_query_vector.index(max(current_query_vector))
                if query_vector[max_idx] == 0.0:
                    query_vector[max_idx] = current_query_vector[max_idx]
                    done = True
                else:
                    current_query_vector[max_idx] = 0.0
            else:
                done = True
    metric = cosine(service_vector, query_vector)
    return metric


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
    print(f'{a} {b} {scores}')
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
    def __init__(self, key:str, path:str, limit:int=0, model:str='dpw', jt:float=.45, 
    lt:int=2, ct:float=.5, st:float=0.05, n:int=5, latent:bool=False, k:int=2, kl:int=0):
        self.services = {}
        self.idx = {}
        self.keywords = None
        self.jt = jt
        self.lt = lt
        self.ct = ct
        self.st = st
        
        if model == 'dpw':
            self.model = dp.DPWModel(corpus=corpus.WebCorpus(key, path, limit=limit), n=n, c=dp.Cutoff.pareto20, latent=latent, k=k)
        else:
            self.model = dp.DPWCModel(corpus=corpus.WebCorpus(key, path, limit=limit), n=n, kl=kl, c=dp.Cutoff.pareto20, latent=latent, k=k)


    def add(self, service):
        key = service['id']
        if key not in self.services:
            self.services[key] = service['description']
            self.model.fit(service['description'])
    

    def buildIdx(self):
        # Add a reverse index to improve Cosine Similarity
        # Get all the list of all the existing keywords
        list_of_keywords = []
        for _,d in self.services.items():
            list_of_keywords.extend(d)
        self.keywords = sorted(list(set(list_of_keywords)))
        # Compute the vectors for the index
        for s,d in self.services.items():
            vector = []
            for k in self.keywords:
                if k in d:
                    vector.append(1.0)
                else:
                    vector.append(0.0)
            self.idx[s] = vector


    def match(self, query):
        scoreJaccardString = []
        scoreJaccardLevenshtein = []
        scoreJaccardSemantic = []
        scoreCosineString = []
        scoreCosineLevenshtein = []
        scoreCosineSemantic = []

        # for all services within the db compute the overall scores
        for s,d in self.services.items():
            scores_string = compute_jaccard_scores(query, d, lambda x,y: Levenshtein.distance(x, y))
            scores_levenshtein = compute_jaccard_scores(query, d, lambda x,y: Levenshtein.distance(x, y), t=self.lt)
            scores_semantic = compute_jaccard_scores(query, d, lambda x,y: self.model.predict(x, y), t=self.st, reverse=True)

            scoreJaccardString.append((s, jaccard_query(query, d, scores_string)))
            scoreJaccardLevenshtein.append((s, jaccard_query(query, d, scores_levenshtein)))
            scoreJaccardSemantic.append((s, jaccard_query(query, d, scores_semantic, reverse=True)))

            scoreCosineString.append((s,compute_cosine_similarity(self.keywords, query, self.idx[s], lambda x,y: 1.0 if x == y else 0.0)))
            scoreCosineLevenshtein.append((s,compute_cosine_similarity(self.keywords, query, self.idx[s], lambda x,y: (max(len(x), len(y)) - Levenshtein.distance(x, y))/max(len(x), len(y)))))
            scoreCosineSemantic.append((s,compute_cosine_similarity(self.keywords, query, self.idx[s], lambda x,y: self.model.predict(x, y))))
            
            #scoreCosineString.append((s, cosine_query(query, d, scores_string)))
            #scoreCosineLevenshtein.append((s, cosine_query(query, d, scores_levenshtein)))
            #scoreCosineSemantic.append((s, cosine_query(query, d, scores_semantic, reverse=True)))
        
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
        scoreCosineSemantic = [(i, s) for (i, s) in scoreCosineSemantic if s >= self.ct]
        scoreCosineSemantic.sort(key=lambda x:x[1], reverse=True)

        return {'jaccard':{'string':scoreJaccardString,'levenshtein':scoreJaccardLevenshtein, 'semantic': scoreJaccardSemantic},
        'cosine':{'string': scoreCosineString, 'levenshtein':scoreCosineLevenshtein, 'semantic':scoreCosineSemantic}}
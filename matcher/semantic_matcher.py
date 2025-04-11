# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


from heapq import nlargest
from matcher.fasttext_matcher import FastText_model
from matcher.glove_matcher import Glove_model
from matcher.word2vec_matcher import Word2Vec_model
from matcher.sbert_matcher import Sbert_model
from matcher.levenshtein_matcher import Levenshtein_model
from matcher.metrics import cosine

import semantic.dp as dp
import semantic.corpus as corpus


def pre_compute_scores(a, b, distance, t=0, reverse=False):
    scores = []
    for i in range(len(a)):
        for j in range(len(b)):
            score = distance(a[i], b[j])
            if score != None:
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


def cosine_query(keywords, query, service_vector, similarity, n_jobs=-1):
    query_vector = [0.0] * len(service_vector)
    # compute scores in parallel
    #scores = Parallel(n_jobs=n_jobs, backend='loky')(delayed(similarity)(q, k) for q in query for k in keywords)
    scores = [similarity(q, k) for q in query for k in keywords]
    # reorganize scores
    for i in range(len(query)):
        for j in range(len(keywords)):
            scores_idx = i * len(keywords) + j
            s = scores[scores_idx]
            if s > query_vector[j]:
                query_vector[j] = s
    # truncate query_vector to only k non-zero values
    threshold = min(nlargest(len(query), query_vector))
    query_vector = [score if score >= threshold else 0.0 for score in query_vector]
    metric = cosine(service_vector, query_vector)
    return metric


class SemanticMathcer():
    def __init__(self, path:str, limit:int=0, model:str='dpw', jt:float=.45, 
                ct:float=.5, st:float=0.05, n:int=5, latent:bool=False, k:int=2, 
                kl:int=0, vector_size=50, window_size=3, pretrained="from_scratch", 
                semantic_training=False, output="results", n_threads=32):
        self.idx = {}
        self.services = {}
        self.jt = jt
        self.ct = ct
        self.st = st
        self.reverse = True
        self.model_name = model
        if model == 'dpw':
            self.model = dp.DPWModel(corpus=corpus.WebCorpus(path, limit=limit), n=n, c=dp.Cutoff.pareto20, latent=latent, k=k)
        elif model == "dpwc":
            self.model = dp.DPWCModel(corpus=corpus.WebCorpus(path, limit=limit), n=n, kl=kl, c=dp.Cutoff.pareto20, latent=latent, k=k)
        elif model == "fasttext":
            self.model = FastText_model(corpus_path=path, vector_size=vector_size, window_size=window_size, output=output, pretrained=pretrained, n_threads=n_threads)
        elif model == "glove":
            self.model = Glove_model(corpus_path=path, vector_size=vector_size, window_size=window_size, output=output, pretrained=pretrained, n_threads=n_threads)
        elif model == "word2vec":
            self.model = Word2Vec_model(corpus_path=path, vector_size=vector_size, window_size=window_size, output=output, pretrained=pretrained, n_threads=n_threads)
        elif model == "sbert":
            self.model = Sbert_model(corpus_path=path, vector_size=vector_size, output=output,
                                      pretrained=pretrained, semantic_training=semantic_training)
        elif model == "string":
            self.model = Levenshtein_model()
            self.reverse = False
        elif model == "levenshtein":
            self.model = Levenshtein_model()
            self.reverse = False

    def add(self, service):
        key = service['id']
        if key not in self.services:
            self.services[key] = service['description']
            if "dpw" in self.model_name: 
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
        
        # DPW and DPWC do this internally
        if hasattr(self.model, 'calculate_bias') and callable(self.model.calculate_bias):
            self.model.calculate_bias(self.keywords)
    
    def match(self, query):
        scoreJaccardSemantic = []
        scoreCosineSemantic = []

        # for all services within the db compute the overall scores
        for s,d in self.services.items():
            
            scores_semantic = pre_compute_scores(query, d, lambda x,y: self.model.predict(x, y), t=self.st, reverse=self.reverse)

            scoreJaccardSemantic.append((s, jaccard_query(query, d, scores_semantic, reverse=self.reverse)))
            
            scoreCosineSemantic.append((s,cosine_query(self.keywords, query, self.idx[s], lambda x,y: self.model.predict(x, y))))

        ## Semantic
        scoreJaccardSemantic = [(i, s) for (i, s) in scoreJaccardSemantic if s >= self.jt]
        scoreJaccardSemantic.sort(key=lambda x:x[1], reverse=self.reverse)

        ## Semantic
        scoreCosineSemantic = [(i, s) for (i, s) in scoreCosineSemantic if s >= self.jt]
        scoreCosineSemantic.sort(key=lambda x:x[1], reverse=self.reverse)

        return {'jaccard':scoreJaccardSemantic, 'cosine':scoreCosineSemantic}
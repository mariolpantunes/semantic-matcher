from gensim.models import Word2Vec
import gensim.downloader as api
import pathlib
import glob
import time
import json

import numpy as np
from matcher.tokenizer import tokenizer


class Word2Vec_model():

    def __init__(self, corpus_path, vector_size, window_size, output, pretrained="from_scratch", n_threads=32) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)

        if pretrained == "pretrained":
            self.model = api.load("word2vec-google-news-300")
        else:
            dataset = pathlib.Path(corpus_path)
            preprocessed_dataset = dataset/"processed_setences.txt"

            self.dataset_preprocessing(dataset, preprocessed_dataset)

            self.model = Word2Vec(json.load(open(preprocessed_dataset)), 
                                vector_size=vector_size,
                                window=window_size,
                                min_count=1, 
                                workers=n_threads, 
                                sg=1, 
                                hs=0, 
                                negative=15, 
                                seed=17)

            self.model.save(str(self.output /'w2v.model'))

            self.model = self.model.wv
        self.bias = None

    def dataset_preprocessing(self, dataset, preprocessed_dataset):
        train_files = glob.glob(str(dataset)+'/*.csv')
        setences_tokens = []

        for f in train_files:
            with open(f, 'rt', newline='', encoding='utf-8') as f:
                snippets = f.readlines()
                for s in snippets:
                    setences_tokens.append(tokenizer(s))
                    
        json.dump(setences_tokens, open(preprocessed_dataset, 'w'))

    def fit(self, text):
        pass

    def predict(self, x, y):

        if x in self.model and y in self.model:
            term_1 = self.model[x]
            term_2 = self.model[y]

            return np.dot(term_1, term_2)/(np.linalg.norm(term_1)*np.linalg.norm(term_2))
        return self.bias

    def calculate_bias(self, list_words):
        res = []

        for i in range(len(list_words)):
            for j in range(i+1, len(list_words)):
                sim = self.predict(list_words[i], list_words[j])
                if sim != None:
                    res.append(sim)
        
        self.bias = sum(res) / len(res)
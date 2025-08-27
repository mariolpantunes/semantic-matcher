import io
import numpy as np
import glob
import pathlib
from matcher.tokenizer import tokenizer
import fasttext
import gzip

class FastText_model():

    def __init__(self, corpus_path, vector_size, window_size, output, pretrained="from_scratch", n_threads=32) -> None:
        self.output = output
        if pretrained == "pretrained":
            fin = io.open(str(self.output / "pretrained.vec"), 'r', encoding='utf-8', newline='\n', errors='ignore')
            fin.readline()
            self.vectors = {}
            for line in fin:
                tokens = line.rstrip().split(' ')
                self.vectors[tokens[0]] = [float(x) for x in  tokens[1:]]
            self.model = None
        elif pretrained == "pretrained_optimized":
            dataset = pathlib.Path(corpus_path)
            preprocessed_dataset = dataset/"processed_setences.txt"
            self.output.mkdir(parents=True, exist_ok=True)

            self.dataset_preprocessing(dataset, preprocessed_dataset)

            self.model = fasttext.train_unsupervised(
                    str(preprocessed_dataset), 
                    'skipgram', 
                    epoch=10,
                    dim=300, 
                    ws=window_size, 
                    minCount=1, 
                    thread=n_threads,
                    pretrainedVectors=str(self.output / "pretrained.vec"))
            
            self.model.save_model(str(self.output/"trained_model.bin"))
        else:
            dataset = pathlib.Path(corpus_path)
            preprocessed_dataset = dataset/"processed_setences.txt"
            self.output.mkdir(parents=True, exist_ok=True)

            self.dataset_preprocessing(dataset, preprocessed_dataset)

            self.model = fasttext.train_unsupervised(
                    str(preprocessed_dataset), 
                    'skipgram', 
                    epoch=10,
                    dim=vector_size, 
                    ws=window_size, 
                    minCount=1, 
                    thread=n_threads)
            
            self.model.save_model(str(self.output/"trained_model.bin"))

        self.bias = None
        
    def dataset_preprocessing(self, dataset, preprocessed_dataset):
        train_files = glob.glob(str(dataset)+'/*.csv.gz')

        # Read the files in the dataset and create setences
        print('Generating tokens from files.')

        # Text Mining Pipeline
        
        aggregated_files = open(preprocessed_dataset, "w")

        for f in train_files:
            with gzip.open(f, mode='rt', newline='', encoding='utf-8') as f:
                snippets = f.readlines()
                for s in snippets:
                    for token in tokenizer(s):
                        aggregated_files.write(token+" ")
            aggregated_files.write("\n")

        aggregated_files.close()

    def fit(self, text):
        pass

    def predict(self, x, y):
        if self.model != None:
            term_1 = self.model.get_word_vector(x)
            term_2 = self.model.get_word_vector(y)
        else:
            if x in self.vectors and y in self.vectors: 
                term_1 = self.vectors[x]
                term_2 = self.vectors[y]
            else:
                return self.bias
        return np.dot(term_1, term_2)/(np.linalg.norm(term_1)*np.linalg.norm(term_2))

    def calculate_bias(self, list_words):
        if self.model == None:
            res = []

            for i in range(len(list_words)):
                for j in range(i+1, len(list_words)):
                    sim = self.predict(list_words[i], list_words[j])
                    if sim != None:
                        res.append(sim)
            
            self.bias = sum(res) / len(res)
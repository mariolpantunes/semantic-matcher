import numpy as np
import glob
import pathlib
from matcher.tokenizer import tokenizer
import subprocess
import gzip

class Glove_model():

    def __init__(self, corpus_path, vector_size, window_size, output, pretrained="from_scratch", n_threads=32) -> None:
        self.output = output

        if pretrained == "pretrained":
            model_file = output/f"glove.6B.{vector_size}d.txt"
        else:
            dataset = pathlib.Path(corpus_path)
            preprocessed_dataset = dataset/"processed_setences.txt"
            self.output.mkdir(parents=True, exist_ok=True)

            self.dataset_preprocessing(dataset, preprocessed_dataset)

            command = f"./glove/vocab_count -min-count 1 -verbose 2 < {preprocessed_dataset} > {self.output/'vocab.txt'}"
            subprocess.run(command, shell=True)
            command = f"./glove/cooccur -vocab-file {self.output/'vocab.txt'} -verbose 2 -window-size {window_size} < {preprocessed_dataset} > {self.output/'cooccurrence.bin'}"
            subprocess.run(command, shell=True)

            command = f"./glove/shuffle -verbose 2 < {self.output/'cooccurrence.bin'} > {self.output/'cooccurrence_shuffle.bin'}"
            subprocess.run(command, shell=True)

            command = f"./glove/glove -save-file {self.output/'vectors'} -threads {n_threads} -input-file {self.output/'cooccurrence_shuffle.bin'} -x-max 10 -iter 20 -vector-size {vector_size} -binary 2 -vocab-file {self.output/'vocab.txt'} -verbose 2"
            subprocess.run(command, shell=True)

            model_file = self.output / 'vectors.txt'
            
        self.bias = None


        with open(model_file, 'r') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                vectors[vals[0]] = [float(x) for x in vals[1:]]
            vocab_size = len(vectors)
            self.vocab = {w: idx for idx, w in enumerate(vectors.keys())}
            ivocab = {idx: w for idx, w in enumerate(vectors.keys())}

            vector_size = len(vectors[ivocab[0]])
            self.W = np.zeros((vocab_size, vector_size))

            for word, v in vectors.items():
                if word == '<unk>':
                    continue
                self.W[self.vocab[word], :] = v

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
        if x in self.vocab and y in self.vocab:
            term_1 = self.W[self.vocab[x]]
            term_2 = self.W[self.vocab[y]]
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
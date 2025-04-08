#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import tqdm
import json
import pickle
import secret
import configs
import logging
import argparse
import time


from matcher.semantic_matcher import SemanticMathcer
from matcher.metrics import mean_average_precision, average_precision
import pathlib


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main(args):

    for config in configs.configs:
        if config["model"] == "sbert":
            semantic = "semtrain" if config["semantic_training"] else "base"
            if config['pretrained'] == 'pretrained' or config['pretrained'] == 'pretrained_optimized':
                output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}_{semantic}"
            else:
                output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}_{semantic}_{config['vector_size']}"
            # create the semantic matcher object
            start_time = time.time()
            semantiMatcher = SemanticMathcer(path=configs.cache, output=output, **config) 
            init_time = time.time() - start_time

        elif config["model"] == "glove":
            if config['pretrained'] == 'pretrained':
                output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}_{config['vector_size']}"
            else:
                output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}_{config['vector_size']}_{config['window_size']}"
            # create the semantic matcher object
            start_time = time.time()
            semantiMatcher = SemanticMathcer(path=configs.cache, output=output, **config) 
            init_time = time.time() - start_time

        elif config["model"] == "word2vec":
            if config['pretrained'] == 'pretrained':
                output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}"
            else:
                output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}_{config['vector_size']}_{config['window_size']}"
            # create the semantic matcher object
            start_time = time.time()
            semantiMatcher = SemanticMathcer(path=configs.cache, output=output, **config) 
            init_time = time.time() - start_time

        elif "dpw" in config["model"]:
            output = pathlib.Path(args.o) / config["model"]
            output.mkdir(parents=True, exist_ok=True)
            # create the semantic matcher object
            start_time = time.time()
            semantiMatcher = SemanticMathcer(path=configs.cache, output=output, **config) 
            init_time = time.time() - start_time

        elif config["model"] == "levenshtein" or config["model"] == "string":
            output = pathlib.Path(args.o) / config["model"]
            output.mkdir(parents=True, exist_ok=True)
            # create the semantic matcher object
            start_time = time.time()
            semantiMatcher = SemanticMathcer(path=configs.cache, **config) 
            init_time = time.time() - start_time
        elif config["model"] == "fasttext":
            if config['pretrained'] != "from_scratch":
                output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}"
            else:
                output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}_{config['vector_size']}_{config['window_size']}"
                # create the semantic matcher object
            start_time = time.time()
            semantiMatcher = SemanticMathcer(path=configs.cache, output=output, **config) 
            init_time = time.time() - start_time

        else:
            output = pathlib.Path(args.o) / config["model"] / f"{config['pretrained']}_{config['vector_size']}_{config['window_size']}"
            # create the semantic matcher object
            start_time = time.time()
            semantiMatcher = SemanticMathcer(path=configs.cache, output=output, **config) 
            init_time = time.time() - start_time

        if "dpw" in config["model"]: 
            # load data model
            if args.l is not None:
                logger.info(f'Load semantic model: {args.l}')
                with open(args.l, 'rb') as input_file:
                    model = pickle.load(input_file)
                    semantiMatcher.model = model

        # load the scenario.json
        with open(args.i) as json_file:
            scenario = json.load(json_file)

        methods = ['jaccard', 'cosine']

        # load the services and register them
        services = scenario['services']
        start_time = time.process_time()
        for s in tqdm.tqdm(services):
            semantiMatcher.add(s)
        train_time = time.process_time()-start_time
        
        semantiMatcher.buildIdx()

        # run the tests
        tests = ['queries m2m', 'queries one-error', 'queries two-errors-one-word',
        'queries two-errors-two-words', 'queries one-synonym', 'queries two-synonyms',
        'queries three-synonyms', 'queries four-synonyms']

        performance = {'jaccard':[], 'cosine':[]}

        output_list = []
        start_time = time.process_time()
        for t in tqdm.tqdm(tests):
            queries = scenario[t]
            for q in tqdm.tqdm(queries, leave=False):
                services = semantiMatcher.match(q['query'])

                # store results for output
                output_list.append({'query':q, 'services':services})

                queryId = int(q['id']) % 100
                relevant = [queryId]
                
                for method in methods:
                    received = [i for i, _ in services[method]]
                    performance[method].append((relevant, received))
        prediction_time = time.process_time()-start_time

        # store the output in a file
        with open(output/ 'output.log', 'w') as outfile:
            for result in output_list:
                json_object = json.dumps(result)
                outfile.write(json_object + '\n')

        with open(output/ 'times.txt', 'w') as outfile:
            train_time += init_time
            outfile.write(f"Training time: {train_time}\n")
            outfile.write(f"Inference time: {prediction_time}\n")
            # compute the mean average precision
            for method in methods:
                temp_list = performance[method]
                relevant, received = [], []
                for rel, rec in temp_list:
                    relevant.append(rel)
                    received.append(rec)
                value = mean_average_precision(relevant, received)
                print(f'{method} = {value}')
                outfile.write(f'{method} = {value}\n')

        # store the semantic model
        if args.s is not None and "dpw" in args.model:
            logger.info(f'Store semantic model: {args.s}')
            with open(args.s, 'wb') as output_file:
                pickle.dump(semantiMatcher.model, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Matcher evaluation tool')
    parser.add_argument('-i', type=str, help='input file', default='scenario.json')
    parser.add_argument('-o', type=str, help='output folder', default="results/")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l', type=str, help='input model')
    group.add_argument('-s', type=str, help='output model')
    args = parser.parse_args()

    main(args)

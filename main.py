#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import tqdm
import json
import config
import logging
import argparse
import numpy as np

from semantic_matcher import SemanticMathcer
from metrics import mean_average_precision, average_precision


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main(args):
    # create the semantic matcher object
    semantiMatcher = SemanticMathcer(config.key, config.cache)

    # load the scenario.json
    with open(args.i) as json_file:
        scenario = json.load(json_file)

    methods = ['jaccard', 'cosine']
    submethods = ['string', 'levenshtein', 'semantic']
    

    # load the services and register them
    services = scenario['services']
    for s in tqdm.tqdm(services):
        semantiMatcher.add(s)

    # run the tests
    tests = ['queries m2m', 'queries one-error', 'queries two-errors-one-word',
    'queries two-errors-two-words', 'queries one-synonym', 'queries two-synonyms',
    'queries three-synonyms', 'queries four-synonyms']

    performance = {'jaccard':{'string':[], 'levenshtein':[], 'semantic':[]}, 
    'cosine':{'string':[], 'levenshtein':[], 'semantic':[]}}

    output_list = []

    for t in tqdm.tqdm(tests):
        queries = scenario[t]
        for q in tqdm.tqdm(queries, leave=False):
            services = semantiMatcher.match(q['query'])

            # store results for output
            output_list.append({'query':q, 'services':services})

            queryId = int(q['id']) % 100
            relevant = [queryId]
            
            for method in methods:
                for submethod in services[method]:
                    received = [i for i, _ in services[method][submethod]]
                    performance[method][submethod].append((relevant, received))

    # compute the mean average precision
    for method in methods:
        for submethod in submethods:
            temp_list = performance[method][submethod]
            relevant, received = [], []
            for rel, rec in temp_list:
                relevant.append(rel)
                received.append(rec)
            value = mean_average_precision(relevant, received)
            print(f'{method}/{submethod} = {value}')
    
    # store the output in a file
    with open('output.json', 'w') as outfile:
        json_object = json.dumps({'results':output_list})
        outfile.write(json_object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Matcher evaluation tool')
    parser.add_argument('-i', type=str, help='input file', default="scenario.json")
    args = parser.parse_args()

    main(args)


#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import json
import logging
import argparse

from semantic_matcher import SemanticMathcer


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def precision (relevant, received):
    """Computes the precision of a reply.
    """
    received_relevants = 0.0
    for item in received:
        if item in relevant:
            received_relevants+=1
    return received_relevants/len(received)


def average_precision(relevant, received):
    """Computes the average precision of a reply.
    """
    result = 0.0
    for k in range(len(received)):
        if received[k] in relevant:
            result+=precision(relevant, received[:k+1])
    return result/len(relevant)


def main(args):
    # create the semantic matcher object
    semantiMatcher = SemanticMathcer()
    
    # load the scenario.json
    with open(args.i) as json_file:
        scenario = json.load(json_file)
        
        # compute the average precision
        methods = ['jaccard', 'cosine']
        submethods = ['string', 'levenshtein']#, 'semantic']

        # load the services and register them
        services = scenario['services']
        for s in services:
            semantiMatcher.add(s)
        
        # run the tests
        tests = ['queries m2m']

        for t in tests:
            queries = scenario[t]
            for q in queries:
                services = semantiMatcher.match(q['query'])

                queryId = int(q['id'])%100
                relevant = [queryId]

                for method in methods:
                    for submethod in services[method]:
                        received = [i for i,_ in services[method][submethod]]
                        value = average_precision(relevant, received)
                        print(f'{method}/{submethod} = {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Matcher evaluation tool')
    parser.add_argument('-i', type=str, help='input file', default="scenario.json")
    args = parser.parse_args()
    
    main(args)
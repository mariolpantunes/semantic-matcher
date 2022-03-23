
#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import json
import config
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from semantic_matcher import SemanticMathcer


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def precision(relevant, received):
    received_relevants = 0.0
    for item in received:
        if item in relevant:
            received_relevants += 1
    return received_relevants/len(received)


def average_precision(relevant, received):
    result = 0.0
    for k in range(len(received)):
        if received[k] in relevant:
            result += precision(relevant, received[:k+1])
    return result/len(relevant)


def main(args):
    # create the semantic matcher object
    semantiMatcher = SemanticMathcer(config.key, config.cache)

    # load the scenario.json
    with open(args.i) as json_file:
        scenario = json.load(json_file)

    # compute the average precision
    variants=["precision","count"]
    methods = ['jaccard', 'cosine']
    submethods =  ['semantic']
    #['string', 'levenshtein',

    # load the services and register them
    services = scenario['services']
    for s in services:
        semantiMatcher.add(s)

    # run the tests
    tests = ['queries m2m', 'queries one-error', 'queries two-errors-one-word',
                'queries two-errors-two-words', 'queries one-synonym', 'queries two-synonyms',
                'queries three-synonyms', 'queries four-synonyms']

    #tests = ['queries two-errors-two-words']

    groups = {0: "M2M", 10: "E2M(1/1)", 11: "E2M(1/2)", 12: "E2M(2/2)",
                20: "U2M(1)", 21: "U2M(2)", 22: "U2M(3)", 23: "U2M(4)"}

    sortedGroups = []
    for key in sorted(groups.keys()):
        sortedGroups += [groups[key]]
    
    columnCount = len(sortedGroups)
    results = {}

    for t in tests:
        queries = scenario[t]
        print(f'Scenario {t}')
        for q in queries:
            services = semantiMatcher.match(q['query'])

            queryId = int(q['id']) % 100
            relevant = [queryId]
            group = groups[int(q['id']/100)]
            groupI = sortedGroups.index(group)

            for variant in variants:
                for method in methods:
                    for submethod in services[method]:
                        print(f'Services = {services}')
                        received = [i for i, _ in services[method][submethod]]
                        print(f'Received = {received}')
                        value = average_precision(relevant, received)
                        print(f'{group}/{method}/{submethod} = {value}')
                        level0 = results.get(variant,{})
                        column = groupI
                        if variant == 'precision':
                            initialPrecision = np.zeros((30,columnCount),dtype=float)
                            level1 = level0.get(method,{})
                            level2 = level1.get(submethod,initialPrecision)
                            row = queryId
                            level2[[row],[column]]=value
                        elif variant == 'count':
                            initialCount = np.zeros((10,columnCount),dtype=float)
                            initialCount[[0]]=1
                            level1 = level0.get(method,{})
                            level2 = level1.get(submethod,initialCount)
                            row = math.ceil(value*10)-1
                            if row < 0:
                                row=0
                            level2[[row],[column]]+=1.0/30
                            level2[[0],[column]]-=1.0/30
                        else:
                            raise Exception("Oops: How did we get here :-)")
                        level1[submethod] = level2
                        level0[method] = level1
                        results[variant] = level0
    #Make Graphics
    pdf = PdfPages('out/heatmaps.pdf')

    # Plot it out
    for variant in variants:
        for method in methods:
            data = []
            for i in range(len(submethods)):
                submethod = submethods[i]
                data = results[variant][method][submethod]

                plt.rc('font', size='9')
                plt.subplot(1,len(submethods),i+1)
                plt.pcolor(data, cmap=plt.cm.Blues, alpha=0.8, vmin=0, vmax=1)
                plt.title(submethod.capitalize())
                # Format
                fig = plt.gcf()
                # get the axis
                ax = plt.gca()
                # turn off the frame
                # ax.set_frame_on(False)
                # put the major ticks at the middle of each cell
                ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
                for t in ax.xaxis.get_major_ticks():
                    t.tick1On = False
                    t.tick2On = False
                for t in ax.yaxis.get_major_ticks():
                    t.tick1On = False
                    t.tick2On = False
                ax.grid(False)
                ax.set_xlim(0,data.shape[1])
                ax.set_xticklabels(sortedGroups, minor=False)
                plt.xticks(rotation=90)

                if i==0:
                    if variant == "precision":
                        plt.ylabel("Services")
                        precisionYLabels = range(data.shape[0])
                        ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
                        ax.set_yticklabels(precisionYLabels, minor=False)
                    elif variant == "count":
                        plt.ylabel("Average Precision")
                        precisionYLabels = np.linspace(0,1,data.shape[0]+1)
                        ax.set_yticks(np.arange(data.shape[0]+1), minor=False)
                        ax.set_yticklabels(precisionYLabels, minor=False)
                else:
                    ax.set_yticks([], minor=False)

                if i==len(submethods)-1:
                    if variant == "precision":
                        plt.colorbar(label="Average Precision")
                        fig.set_size_inches(6,6)
                    if variant == "count":
                        plt.colorbar(label="Occurrences")
                        fig.set_size_inches(6,3)

            #fig.suptitle(method.capitalize())
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            fig.tight_layout()
            pdf.savefig()
            plt.close()

    pdf.close()

    pdf = PdfPages('out/boxplot.pdf')
    medianprops = dict(linewidth=4)
    for method in methods:
        data = []
        for i in range(len(submethods)):
            submethod = submethods[i]
            data = results["precision"][method][submethod]
            plt.subplot(1,len(submethods),i+1)
            plt.rc('font', size='9')
            plt.boxplot(data, labels = sortedGroups, medianprops = medianprops, whis=[5,95])
            plt.title(submethod.capitalize())
            fig = plt.gcf()
            ax = plt.gca()
            plt.xticks(rotation=90)
            if i==0:
                plt.ylabel("Average Precision")
            if i==len(submethods)-1:
                fig.set_size_inches(6,3)
            
        #fig.suptitle(method.capitalize())
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig.tight_layout()
        pdf.savefig()
        plt.close()

    pdf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Semantic Matcher evaluation tool')
    parser.add_argument('-i', type=str, help='input file',
                        default="scenario.json")
    args = parser.parse_args()

    main(args)

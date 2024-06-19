#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matcher.metrics import average_precision, mean_average_precision


def main(args):
    # load the output.json
    with open(args.i) as log_file:
        output = log_file.readlines()

    variants = ['precision', 'count']
    methods = ['jaccard', 'cosine']
    submethods = ['string', 'levenshtein', 'semantic']
    
    groups = {0: "M2M", 10: "E2M(1/1)", 11: "E2M(1/2)", 12: "E2M(2/2)",
    20: "U2M(1)", 21: "U2M(2)", 22: "U2M(3)", 23: "U2M(4)"}

    sortedGroups = []
    for key in sorted(groups.keys()):
        sortedGroups += [groups[key]]
    
    columnCount = len(sortedGroups)
    results = {}
    performance = {'jaccard':{'string':{}, 'levenshtein':{}, 'semantic':{}}, 
    'cosine':{'string':{}, 'levenshtein':{}, 'semantic':{}}}

    for line in output:
        row = json.loads(line)
        q = row['query']
        services = row['services']
        queryId = int(q['id']) % 100
        relevant = [queryId]
        group = groups[int(q['id']/100)]
        groupI = sortedGroups.index(group)

        for variant in variants:
            for method in methods:
                for submethod in services[method]:
                    received = [i for i, _ in services[method][submethod]]
                    value = average_precision(relevant, received)
                    #print(f'{group}/{method}/{submethod} = {value}')
                    if group not in performance[method][submethod]:
                        performance[method][submethod][group] = []
                    performance[method][submethod][group].append((relevant, received))

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
            plt.boxplot(data, tick_labels = sortedGroups, medianprops = medianprops, whis=[5,95])
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


    # compute the mean average precision
    for method in methods:
        for submethod in submethods:
            for group_key in groups:
                temp_list = performance[method][submethod][groups[group_key]]
                relevant, received = [], []
                for rel, rec in temp_list:
                    relevant.append(rel)
                    received.append(rec)
                value = mean_average_precision(relevant, received)
                print(f'{method}/{submethod}/{groups[group_key]} = {value}')
    
    # global mPA
    for method in methods:
        for submethod in submethods:
            relevant, received = [], []
            for group_key in groups:
                temp_list = performance[method][submethod][groups[group_key]]
                for rel, rec in temp_list:
                    relevant.append(rel)
                    received.append(rec)
            value = mean_average_precision(relevant, received)
            print(f'{method}/{submethod} = {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Matcher evaluation tool')
    parser.add_argument('-i', type=str, help='input file', default="output.log")
    args = parser.parse_args()

    main(args)
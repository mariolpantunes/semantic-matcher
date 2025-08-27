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
import os

def main(args):
    calculation_variants = ['precision', 'count']
    distance_methods = ['jaccard', 'cosine']
    query_groups = {0: "M2M", 10: "E2M(1/1)", 11: "E2M(1/2)", 12: "E2M(2/2)",
    20: "U2M(1)", 21: "U2M(2)", 22: "U2M(3)", 23: "U2M(4)"}
    sorted_query_groups = []
    for key in sorted(query_groups.keys()):
        sorted_query_groups += [query_groups[key]]

    similarity_models = []

    for folder in os.listdir(args.d):
        folder_path = os.path.join(args.d, folder)
        if os.path.isdir(folder_path):
            # Get subfolders
            subfolders = [
                os.path.join(folder, sub)
                for sub in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, sub))
            ]

            if subfolders:
                similarity_models.extend(subfolders)
            else:
                similarity_models.append(folder)

    pretrained_models = [m for m in similarity_models if "pretrained" in m]
    from_scratch_models = [m for m in similarity_models if "pretrained" not in m]

    columnCount = len(sorted_query_groups)
    results = {}
    performance = {d:{m : {} for m in similarity_models} for d in distance_methods}

    for model in similarity_models:
        output_file = os.path.join(args.d, model, "output.log")

        for line in open(output_file, "r").readlines():
            row = json.loads(line)
            q = row['query']
            services = row['services']
            queryId = int(q['id']) % 100
            relevant = [queryId]
            group = query_groups[int(q['id']/100)]
            groupI = sorted_query_groups.index(group)

            for variant in calculation_variants:
                for method in distance_methods:
                    received = [i for i, _ in services[method]]
                    value = round(average_precision(relevant, received),2)

                    if group not in performance[method][model]:
                        performance[method][model][group] = []
                    performance[method][model][group].append((relevant, received))

                    level0 = results.get(variant,{})
                    column = groupI
                    if variant == 'precision':
                        initialPrecision = np.zeros((30,columnCount),dtype=float)
                        level1 = level0.get(method,{})
                        level2 = level1.get(model,initialPrecision)
                        row = queryId
                        level2[[row],[column]]=value
                    elif variant == 'count':
                        initialCount = np.zeros((10,columnCount),dtype=float)
                        initialCount[[0]]=1
                        level1 = level0.get(method,{})
                        level2 = level1.get(model,initialCount)
                        row = math.ceil(value*10)-1
                        if row < 0:
                            row=0
                        level2[[row],[column]]+=1.0/30
                        level2[[0],[column]]-=1.0/30
                    else:
                        raise Exception("Oops: How did we get here :-)")
                    level1[model] = level2
                    level0[method] = level1
                    results[variant] = level0

    #Make Graphics
    pdf = PdfPages(os.path.join(args.d,'heatmaps.pdf'))

    # Plot it out
    for variant in calculation_variants:
        for method in distance_methods:
            data = []
            for i in range(len(from_scratch_models)):
                model = from_scratch_models[i]
                data = results[variant][method][model]

                plt.rc('font', size='9')
                plt.subplot(1,len(from_scratch_models),i+1)
                plt.pcolor(data, cmap=plt.cm.Blues, alpha=0.8, vmin=0, vmax=1)
                plt.title(model.split("/")[0].capitalize())
                fig = plt.gcf()
                ax = plt.gca()

                ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
                for t in ax.xaxis.get_major_ticks():
                    t.tick1On = False
                    t.tick2On = False
                for t in ax.yaxis.get_major_ticks():
                    t.tick1On = False
                    t.tick2On = False
                
                ax.grid(False)
                ax.set_xlim(0,data.shape[1])
                ax.set_xticklabels(sorted_query_groups, minor=False)
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

                if i==len(from_scratch_models)-1:
                    if variant == "precision":
                        plt.colorbar(label="Average Precision")
                        fig.set_size_inches(16,6)
                    if variant == "count":
                        plt.colorbar(label="Occurrences")
                        fig.set_size_inches(16,6)

            #fig.suptitle(method.capitalize())
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            fig.tight_layout()
            pdf.savefig()
            plt.close()

    pdf.close()

    pdf = PdfPages(os.path.join(args.d,'boxplot.pdf'))
    medianprops = dict(linewidth=4)
    for method in distance_methods:
        data = []
        for i in range(len(from_scratch_models)):
            model = from_scratch_models[i]
            data = results["precision"][method][model]
            plt.subplot(1,len(from_scratch_models),i+1)
            plt.rc('font', size='9')
            plt.boxplot(data, labels = sorted_query_groups, medianprops = medianprops, whis=[5,95])
            plt.title(model.split("/")[0].capitalize())
            fig = plt.gcf()
            ax = plt.gca()
            plt.xticks(rotation=90)
            if i==0:
                plt.ylabel("Average Precision")
            if i==len(from_scratch_models)-1:
                fig.set_size_inches(16,6)
            
        #fig.suptitle(method.capitalize())
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig.tight_layout()
        pdf.savefig()
        plt.close()

    pdf.close()

    # compute the mean average precision
    
    print("\n\n Mean Average precision per query")
    print("\n\n Pretrained")
    header = " | ".join(pretrained_models)

    for method in pretrained_models:
        print(f"\n ----------------{method}-------------")
        print(f"Query | {header} |")
        for group_key in query_groups:
            row = f"| {query_groups[group_key]} | "
            for model in similarity_models:
                relevant, received = [], []
                for rel, rec in performance[method][model][query_groups[group_key]]:
                    relevant.append(rel)
                    received.append(rec)
                value = mean_average_precision(relevant, received)
                row += f"{value:.2f} | "
            print(row)

    print("\n\n From scratch")
    header = " | ".join(from_scratch_models)

    for method in from_scratch_models:
        print(f"\n ----------------{method}-------------")
        print(f"Query | {header} |")
        for group_key in query_groups:
            row = f"| {query_groups[group_key]} | "
            for model in similarity_models:
                relevant, received = [], []
                for rel, rec in performance[method][model][query_groups[group_key]]:
                    relevant.append(rel)
                    received.append(rec)
                value = mean_average_precision(relevant, received)
                row += f"{value:.2f} | "
            print(row)
    
    # global mPA
    print("\n\n Global Mean Average precision")

    print("\n\n Pretrained")
    header = " | ".join(pretrained_models)
    for method in from_scratch_models:
        print(f"\n ----------------{method}-------------")
        print(f"| Model | mAP |")
        for model in similarity_models:
            relevant, received = [], []
            for group_key in query_groups:
                for rel, rec in performance[method][model][query_groups[group_key]]:
                    relevant.append(rel)
                    received.append(rec)
            value = mean_average_precision(relevant, received)
            print(f'| {model} | {value:.2f} |')
        print()

    print("\n\n From scratch")
    header = " | ".join(from_scratch_models)
    for method in from_scratch_models:
        print(f"\n ----------------{method}-------------")
        print(f"| Model | mAP |")
        for model in similarity_models:
            relevant, received = [], []
            for group_key in query_groups:
                for rel, rec in performance[method][model][query_groups[group_key]]:
                    relevant.append(rel)
                    received.append(rec)
            value = mean_average_precision(relevant, received)
            print(f'| {model} | {value:.2f} |')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Matcher evaluation tool')
    parser.add_argument('-d', type=str, help='input folder', default="results")
    args = parser.parse_args()

    main(args)
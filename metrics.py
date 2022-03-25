# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

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


def mean_average_precision(relevant, received):
    rv = 0.0
    for i in range(len(relevant)):
        rv += average_precision(relevant[i], received[i])
    return rv/len(relevant)

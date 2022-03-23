import statistics
from tkinter import N

from numpy import double
def maxi(arr):
    for i in arr:
        if type(i) not in [int, float]:
            return None
    return max(arr)
def mini(arr):
    for i in arr:
        if type(i) not in [int, float]:
            return None
    return min(arr)
def mediani(arr):
    for i in arr:
        if type(i) not in [int, float]:
            return None
    return statistics.median(arr)
def quantilesi(arr):
    for i in arr:
        if type(i) not in [int, float]:
            return None
    return statistics.quantiles(arr)
def meani(arr):
    for i in arr:
        if type(i) not in [int, float]:
            return None
    return statistics.mean(arr)

def summary(data):
    '''
    Returns R-like summary on data
    '''
    N = len(data)
    m = len(data[0])
    ans = {}
    ans['max'] = [maxi([data[i][j] for i in range(N)]) for j in range(m)]
    ans['min'] = [mini([data[i][j] for i in range(N)]) for j in range(m)]
    ans['median'] = [mediani([data[i][j] for i in range(N)]) for j in range(m)]
    ans['mean'] = [meani([data[i][j] for i in range(N)]) for j in range(m)]
    ans['quantiles'] = [quantilesi([data[i][j] for i in range(N)]) for j in range(m)]
    return ans
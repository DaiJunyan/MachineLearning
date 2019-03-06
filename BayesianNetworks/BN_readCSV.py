"""
This is the new one which can read data from CSV files
CPS4902 Machine Learning ---- Bayesian Network(implement DAG and CPT)
Junyan Dai, Yuefei Chen
1025584, 1025573
Instructor: Dr. Jenny Li
12/21/2018
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as N
import pandas as pd
from collections import defaultdict


def marginal_distribution(X, u):
    """
    Return the marginal distribution for the u'th features of the data points, X.
    """
    values = defaultdict(float)
    elements, counts = N.unique(X[X.columns[u]], return_counts=True)
    for i in range(len(elements)):
        values[elements[i]] = counts[i]/N.sum(counts)

    return values


def marginal_pair_distribution(X, u, v):
    """
    Return the marginal distribution for the u'th and v'th features of the data points, X.
    """
    if u > v:
        u, v = v, u
    values = defaultdict(float)
    x = X.groupby([X.columns[u], X.columns[v]]).size().reset_index(name='Freq')
    for i in range(len(x['Freq'])):
        values[(x[x.columns[0]][i], x[x.columns[1]][i])] = x['Freq'][i]/x['Freq'].sum()

    return values


def calculate_mutual_information(X, u, v):
    """
    X are the data points.
    u and v are the indices of the features to calculate the mutual information for.
    """
    if u > v:
        u, v = v, u
    marginal_u = marginal_distribution(X, u)
    marginal_v = marginal_distribution(X, v)
    marginal_uv = marginal_pair_distribution(X, u, v)
    I = 0.
    for x_u, p_x_u in marginal_u.items():
        for x_v, p_x_v in marginal_v.items():
            if (x_u, x_v) in marginal_uv:
                p_x_uv = marginal_uv[(x_u, x_v)]
                I += p_x_uv * (N.log(p_x_uv) - N.log(p_x_u) - N.log(p_x_v))
    return I


def build_chow_liu_tree(X, n):
    """
    Build a Chow-Liu tree from the data, X. n is the number of features. The weight on each edge is
    the negative of the mutual information between those features. The tree is returned as a networkx
    object.
    """
    G = nx.Graph()
    for v in range(n):
        G.add_node(X.columns[v])
        for u in range(v):
            G.add_edge(X.columns[u], X.columns[v], weight=calculate_mutual_information(X, u, v))
    T = nx.maximum_spanning_tree(G)
    return T


def build_cpt(dag,X):
    adjacency = dag.succ
    #print(adjacency)

    #print the first CPT for x0
    marginal_x0 = marginal_distribution(X,0)
    print(X.columns[0])
    for x0,p_x0 in marginal_x0.items():
        print("%s\t%-7.5f" % (x0, p_x0))

    #The CPTs for other nodes
    for i in range(len(adjacency)):
        if adjacency[X.columns[i]]!={}:
            for j in adjacency[X.columns[i]].items():
                print("\n%s\\%s"%(X.columns[i],j[0]),end="")
                marginal_xi = marginal_distribution(X,i)
                marginal_xj = marginal_distribution(X,X.columns.get_loc(j[0]))
                marginal_xi_xj = marginal_pair_distribution(X,i,X.columns.get_loc(j[0]))
                for xj, p_xj in marginal_xj.items():
                    print("\t%s"%xj,end="")
                for xi,p_xi in marginal_xi.items():
                    print("\n%s"%xi,end="")
                    for xj, p_xj in marginal_xj.items():
                        if (xi,xj) in marginal_xi_xj:
                            p_xi_xj = marginal_xi_xj[(xi,xj)]/p_xi
                        else:
                            p_xi_xj = 1/((p_xi*len(X))+len(marginal_xi))
                        print("\t%-7.5f" % p_xi_xj,end="")


if __name__ == '__main__':
    # data
    df = pd.read_csv('dataSamples4BayesianNetworks.csv')

    n = len(df.columns)  # number of nodes
    T = build_chow_liu_tree(df, n)
    print(T.edges(data=True))
    # add  directions
    DAG = nx.DiGraph()
    DAG.add_edges_from(T.edges)
    # show DAG
    nx.draw(DAG,pos=nx.spring_layout(DAG), with_labels=True, font_weight='bold')
    plt.savefig("DAG4csv.png")
    # print CPTs
    build_cpt(DAG,df)

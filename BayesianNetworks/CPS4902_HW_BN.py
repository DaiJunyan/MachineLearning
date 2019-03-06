"""
CPS4902 Machine Learning ---- Bayesian Network(implement DAG and CPT)
Junyan Dai, Yuefei Chen
1025584, 1025573
Instructor: Dr. Jenny Li
12/13/2018
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as N
from collections import defaultdict


def marginal_distribution(X, u):
    """
    Return the marginal distribution for the u'th features of the data points, X.
    """
    values = defaultdict(float)
    s = 1. / len(X)
    for x in X:
        values[x[u]] += s
    return values


def marginal_pair_distribution(X, u, v):
    """
    Return the marginal distribution for the u'th and v'th features of the data points, X.
    """
    if u > v:
        u, v = v, u
    values = defaultdict(float)
    s = 1. / len(X)
    for x in X:
        values[(x[u], x[v])] += s
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
        G.add_node(v)
        for u in range(v):
            G.add_edge(u, v, weight=calculate_mutual_information(X, u, v))
    T = nx.maximum_spanning_tree(G)
    return T


def build_cpt(dag,X):
    adjacency = dag.succ
    #print(adjacency)

    #print the first CPT for x0
    marginal_x0 = marginal_distribution(X,0)
    print("x0")
    for x0,p_x0 in marginal_x0.items():
        print("%s\t%-7.5f" % (x0, p_x0))

    #The CPTs for other nodes
    for i in range(len(adjacency)):
        if adjacency[i]!={}:
            for j in adjacency[i].items():
                print("\nx%d\\x%d"%(i,j[0]),end="")
                marginal_xi = marginal_distribution(X,i)
                marginal_xj = marginal_distribution(X,j[0])
                marginal_xi_xj = marginal_pair_distribution(X,i,j[0])
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
    X = ['1111',
         '1111',
         '0111',
         '1111',
         '0000',
         '0000',
         '0000']
    n = len(X[0])
    T = build_chow_liu_tree(X, n)
    print(T.edges(data=True))
    # add  directions
    DAG = nx.DiGraph()
    DAG.add_edges_from(T.edges)
    # show DAG
    nx.draw(DAG, with_labels=True, font_weight='bold')
    plt.show()
    # print CPTs
    build_cpt(DAG,X)

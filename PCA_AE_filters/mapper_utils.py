#Data processing
import pandas as pd
import numpy as np
from functools import partial

#For mapper
from scipy.stats import entropy
import gtda.plotting
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection, Eccentricity, Entropy,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph
)

#For filter functions
from sklearn.decomposition import PCA

#Clusterer
from sklearn.cluster import DBSCAN

rs = 1

def get_entropy(node_elements, X, y):
    '''
    Calculate the entropy with respect to y for each node, and the size of each node
    '''
    xy = pd.concat((X,y),axis=1)
    
    entropies = []
    sizes = []
    for node in node_elements: #Extract label for each data point for each node and calculate the probability (proportion) 
        #in the node of each observation
        labels = [xy.iloc[node[i]]['label'] for i in range(0,len(node))]
        value,counts = np.unique(labels, return_counts=True)
        node_entropy = entropy(pk = counts/len(labels))
        entropies.append(node_entropy)
        sizes.append(len(node))
    
    return entropies, sizes

def call_mapper(X, y, rs=rs, eps = 0.25, min_samples = 5, n_intervals = 10, overlap_frac = 0.25, lense = None):
    '''
    Build a mapper with given hyperparameters and return the graph object, the datapoints at each node, 
    the node id's and the entropies and sizes.
    '''
    xy = pd.concat((X,y),axis=1)
    
    filter_func = lense


    # Define cover
    cover = CubicalCover(n_intervals=n_intervals, overlap_frac = overlap_frac)
    
    # Choose clustering algorithm
    clusterer = DBSCAN(eps = eps, min_samples = min_samples)

    # Configure parallelism of clustering step
    n_jobs = -1

    # Initialise pipeline
    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clustering_preprocessing= None,
        clusterer=clusterer,
        graph_step=True,
        verbose=True,
        n_jobs=n_jobs,
        contract_nodes = False
    )
    
    graph = pipe.fit_transform(xy.drop('label', axis=1))
    node_elements = graph.vs["node_elements"]
    node_ids = graph.vs["pullback_set_label"] 
    
    entropies, sizes = get_entropy(node_elements, X, y)
    
        
    return graph, node_elements, node_ids, entropies, sizes

def get_mappers(param_grid, X_m, y_m, rs=1, lense = 'None'):
    '''
    Build all the mappers for a given grid of hyperparameter combinations (one mapper per each combination in the grid)
    Returns a dataframe with the relevant information of the experiment (average size of the nodes of each mapper, average entropy,
    hyperparameter values, etc.)
    '''
    #Hyperparams
    epsilon = []
    minsamples = []
    interval_number = []
    overlap_fractions = []
    #Entropy
    avg_entropy = []
    std_entropy = []
    #Node sizes (in general)
    avg_node_sizes = []
    std_node_sizes = []

    for grid in param_grid:
    
        graph, node_elements, node_ids, entropies, sizes = call_mapper(X = X_m, y = y_m, rs = 1, lense = lense, **grid)
    
        epsilon.append(grid['eps'])
        minsamples.append(grid['min_samples'])
        interval_number.append(grid['n_intervals'])
        overlap_fractions.append(grid['overlap_frac'])
    
        avg_entropy.append(np.mean(entropies))
        std_entropy.append(np.std(entropies))
    
        avg_node_sizes.append(np.mean(sizes))
        std_node_sizes.append(np.std(sizes))

    Mapper_Info = dict()
    Mapper_Info['Epsilon'] = epsilon
    Mapper_Info['MINSamples'] = minsamples
    Mapper_Info['Number of Intervals'] = interval_number
    Mapper_Info['Percentage of Overlap'] = overlap_fractions
    Mapper_Info['Average Node Entropy'] = avg_entropy
    Mapper_Info['Entropy Standard Deviation'] = std_entropy
    Mapper_Info['Average Node Size'] = avg_node_sizes
    Mapper_Info['Standard Deviation of Node Size'] = std_node_sizes

    Mapper_df = pd.DataFrame.from_dict(Mapper_Info)

    return Mapper_df
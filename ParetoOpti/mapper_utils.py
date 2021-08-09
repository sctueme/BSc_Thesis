#Data processing
import pandas as pd
import numpy as np

#For mapper
import gtda.plotting
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection, Eccentricity,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph
)

#For filter functions
from sklearn.decomposition import PCA

#Clusterer
from sklearn.cluster import DBSCAN

#For optimization
from scipy.stats import entropy
import optuna
from functools import partial
from optuna.samplers import TPESampler

rs = 1

def get_variance(node_elements, X, y):
    '''
    Calculate the size and variance of each node for a continuos variable y
    '''

    xy = pd.concat((X,y),axis=1)
    
    variances = []
    sizes = []
    for node in node_elements: #Extract label for each data point for each node and calculate the variance in the node
        labels = [xy.iloc[node[i]]['label'] for i in range(0,len(node))]
        variances.append(np.var(labels))
        sizes.append(len(node))
    
    return variances, sizes


def get_entropy(node_elements, X, y):
    '''
    Calculate the size and entropy of each node with respect to a categorical variable y 
    '''
    xy = pd.concat((X,y),axis=1)
    
    entropies = []
    sizes = []
    for node in node_elements: #Extract label for each data point for each node and calculate the entropy of the node
        labels = [xy.iloc[node[i]]['label'] for i in range(0,len(node))]
        value,counts = np.unique(labels, return_counts=True)
        node_entropy = entropy(pk = counts/len(labels))
        entropies.append(node_entropy)
        sizes.append(len(node))
    
    return entropies, sizes

def get_heterogeneity(node_elements, X, y, h_fn = 'entropy'):

    if h_fn == 'variance':
        heterogeneities, sizes = get_variance(node_elements = node_elements, X = X, y = y)

    else:
        heterogeneities, sizes = get_entropy(node_elements = node_elements, X = X, y = y)


    return heterogeneities, sizes

def call_mapper(X, y, rs=rs, 
                n_intervals = 10, 
                overlap_frac = 0.25, 
                eps = 0.25, 
                min_samples = 3,
                lense = None, 
                h_fn = 'entropy'):
    '''
    Build a single mapper with given hyperparams and return the graph object, the nodes and their datapoints, the heterogeneity
    of each node and the size of each node 
    '''


    xy = pd.concat((X,y),axis=1)
    
    filter_func = lense

    # Define cover
    cover = CubicalCover(n_intervals=n_intervals, overlap_frac = overlap_frac)
    
    # Choose clustering algorithm
    clusterer = DBSCAN(eps = eps, min_samples = min_samples)
    # Configure parallelism of clustering step
    n_jobs = 1

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
    
    heterogeneities, sizes = get_heterogeneity(node_elements = node_elements, X = X, y = y, h_fn = h_fn)
        
    return graph, node_elements, node_ids, heterogeneities, sizes

def objective_mapper(trial, X, y, lense = None, rs = rs, h_fn = 'entropy'):

    n_intervals = trial.suggest_int('n_intervals', 10, 100)
    overlap_frac = trial.suggest_uniform('overlap_frac',0.05,0.5)
    eps = trial.suggest_uniform('eps',0.001,10)
    min_samples = trial.suggest_int('min_samples', 1, 5)
    #Fitting the mapper
    graph, node_elements, node_ids, heterogeneities, sizes = call_mapper(X, y, rs=rs, 
                                                                n_intervals = n_intervals,
                                                                overlap_frac = overlap_frac, 
                                                                eps = eps,
                                                                min_samples = min_samples, 
                                                                lense = lense, 
                                                                h_fn = h_fn)
    
    s = np.mean(sizes)
    all_labels = y.values
    h = np.mean(heterogeneities)

    if h_fn == 'variance':
        all_heterogeneity = np.var(all_labels)
        H = h/all_heterogeneity

    else:
        all_value, all_counts = np.unique(all_labels, return_counts=True)
        all_heterogeneity = entropy(pk = all_counts/len(all_labels))
        H = h/all_heterogeneity


    objective = s, h

    return objective

def optimize_mapper(X, y, objective_fn = objective_mapper, n_trials = 15, timeout = 120, \
    random_state = rs, lense = None, h_fn = 'entropy'):
    sampler = TPESampler(seed=random_state)  # Make the sampler trials reproducible.
    #Maximize size and minimize heterogeneity
    study_mapper = optuna.create_study(directions = ['maximize', 'minimize'], sampler = sampler)
    optimize_mapper = partial(objective_fn, X=X, y=y, lense=lense, h_fn=h_fn)
    study_mapper.optimize(optimize_mapper, n_trials = n_trials, timeout = timeout)

    return study_mapper
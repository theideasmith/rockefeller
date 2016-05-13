import dimensions_kit as dk
import numpy as np
import matplotlib.pyplot as plt
import NetworkIOStreams as nio
import RTRL as network
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import scipy.cluster.hierarchy as sch
# TODO: Investigate stochastic weight training instead of rudimentary gradient descent?

class DataAggregator:
    def __init__(self, network, niterations):
        """
        Solution Analyser is initialized with a network
        object.

        It will take care of running the network
        under different stochastic input conditions
        and present the resulting solutions (in the
        form of weight matrices, but we can get
        arbitrarily complex in the future.

        """
        self.t = 0
        self.niterations = niterations if niterations else 1000000
        self.network = network
        # List, so that we can store
        # arbitrarily large histories
        self.nethist = [ ];

    def iter(self):
        """
        Stores the current network state
        in our history of states
        """
        state = self.network.currentState()
        self.nethist.append(state)

    def collect(self):
        while self.t < self.niterations:
            self.network.step()
            self.iter()
            self.t+=1
        return self

    def getSolution(self):
        return self.network.weights

class MonteCarloKMeans():
    def __init__(self, data, distance):
        """
        data: list of data objects, can be arbitrary
        distance: function for computing distance
           between two datapoints. Can also be arbitrary
        """
        raise NotImplementedError()
    def centroids():
        raise NotImplementedError()


class StochasticSolutionGenerator:

    def __init__(self, aggregators):
        self.dataaggregators = aggregators
        self.solutions = map(
                lambda da:
                da.collect()
                .getSolution(),
                self.dataaggregators)
        self.numNets = len(aggregators)


    def distance(self, solA, solB):
        """
        Compute the matrix norm for matrices solA and solB. 
        You do this to compute an adjacency matrix for 
        the weights, and then cluster the adjacency
        matrix by pairwise similarity. I'll start
        out by seeing what the clusters look like. 

        TODO: We should be doing the random parameters with Monte Carlo 
        sampling, so we get a good distribution over the entire 
        possible input space. 

        This is the first time I am doing this. 
        Matrix norms: 
        http://www.personal.soton.ac.uk/jav/soton/HELM/workbooks/workbook_30/30_4_matrx_norms.pdf 
        """
        delta = np.abs(solB - solA)
        norm = np.sqrt(np.sum(delta**2))
        return -norm
    
    def genCorrelationMatrix(self):
        matrix = np.zeros((self.numNets, self.numNets))
        for i in xrange(self.numNets):
            for j in xrange(self.numNets):
                matrix[i,j] = self.distance(self.solutions[i], self.solutions[j])
        return matrix


    def monte_carlo_aggregate_k_means(self):
        """
        Performs K-means on random
        cluster sizes from a distribution
        and then stops when the clusters
        stop changing significantly
        """
        raise NotImplementedError()
    def k_means(self):
        """
        Naive self implementation of the K-means algorithm
        """
        raise NotImplementedError()


      






if __name__ == "__main__":

    numNets = 1000
    nNodes = 6
    net = network.RTRLNetwork;
   
    params = [(6, 2, 0.5,100)]*numNets
    networks = map(new_aggregator, params)
    solutions = StochasticSolutionGenerator(networks)
    correlations = solutions.genCorrelationMatrix()

    Z = sch.linkage(correlations, method='centroid')
    indexs = sch.leaves_list(Z)

    clustered_correl = correlations[indexs][:,indexs]
    ax = plt.figure().add_subplot(111)
    ax.pcolormesh(clustered_correl)

    #graph = plt.figure(3)
    #G = nx.from_numpy_matrix(clustered_correl)
    #nx.draw_spring(G)
    plt.show()

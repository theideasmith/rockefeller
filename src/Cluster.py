import numpy as np
import time
import NetworkIOStreams as nio
# TODO: Investigate stochastic weight training instead of rudimentary gradient descent?

class RunNetwork:
    def __init__(self, network):
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
        self.network = network
        # List, so that we can store
        # arbitrarily large histories
        self.nethist = [ ];
        self.error = np.zeros(1000)
        self.THRESHOLD = 5.0*10**(-5)  



    def iter(self):
        """
        Stores the current network state
        in our history of states
        """
        state = self.network.currentState()
        self.nethist.append(state)
        self.error[:-1] = self.error[1:];
        self.error[-1] = self.network.error()

    def is_trained(self):
        # print "Error derivative, mean:"
    	# print np.diff(self.error[-10:-1])
	# print np.mean(np.diff(self.error[-10:-1]))
        # print "Error: {0}".format(self.error[-1])
    	training_status = np.mean(self.error[-10:-1])
        # print "Training: {0}, threshold: {1}".format(training_status,self.THRESHOLD)
        if self.t > 500:                                     
            if training_status <= self.THRESHOLD:
                return True

        return False

    def collect(self,log=False):
        print "Log: {0}".format(log)
        while self.is_trained() == False:
            self.network.step()
            if log==True:
                print self.network.error()
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

def ME(solA, solB):                                                                
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
    return (-1)*norm                                                                               
                                                                                               
def genCorrelationMatrix(objects, distanceMetric=ME):                                          
    numNets = len(objects)
    matrix = np.zeros((numNets, numNets))                                            
    for i in xrange(numNets):                                                             
        for j in xrange(numNets):                                                         
            matrix[i,j] = ME(objects[i],objects[j])                  
    return matrix                                                                              


class StochasticSolutionGenerator:

    def __init__(self, networkrunners):
        self.networkrunners = networkrunners
        self.solutions = []
        self.numNets = len(networkrunners)
    	

    def solve(self):
	self.solutions=0
        for r in xrange(len(self.networkrunners)):               
    	    run = self.networkrunners[r]                         
    	    print "Running network: {0}".format(r)               
    	    self.solutions.append(run.collect().getSolution()) 

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


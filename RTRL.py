"""
Real Time Recurrent Learning

Reference
---------
.. [WilliamsZipser1989] `A Learning Algorithm for Continually Running Fully Recurrent neural Networks, Neural Computation 1, 270-280 (1989)
    <http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=6795228&abstractAccess=no&userType=inst>`_
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np
from Network import Network;
from NetworkIOStreams import NetworkIOStream;


class RTRLNetwork(Network):
  """Real Time Recurrent Learning Network"""
  
  def __init__(self, nNodes, io = NetworkIOStream(), eta = 0.001, reset = None):
    """Initialize network
    
    Arguments:
      nNodes (int): number of network nodes
    """
    
    Network.__init__(self, nNodes, io);
    self.eta = eta;
    self.resetNetwork(); 
    self.reset = reset;
    self.resetCounter = 0;
    
    self.constrainWeights();
  
  
  def resetNetwork(self):
    """Reset the network p's to initial state"""
    
    self.p = np.zeros((self.nNodes, self.nNodes * (self.nNodes + self.nInputs))); # p[k, (ij)] = p[k, i*(n+m) + j]
    self.t = 0;
        
    
  
  
  def f(self, x):
    return (1./(1. + np.exp(-x)));
  
  def step(self):
    """Evaluate time step and change weights"""
    
    if not self.reset is None:
      self.resetCounter += 1;
      if self.resetCounter == self.reset:
        self.resetNetwork();
        self.resetCounter = 0;
        #print 'reset'
    
    n = self.nNodes;
    l = self.nNodes + self.nInputs;
    
    # generate input and output
    self.input  = self.io.getInput();
    self.goal   = self.io.getOutput();
    
    # update states
    z = np.hstack((self.state, self.input));
    y = self.f(np.dot(self.weights,  z));
    
    self.state = y;
    self.output = y[self.outputNodes];
    
    
    # update propagation of derivatives w.r.t. weights
    df = y * (1- y);
    
    # input
    ii = np.zeros((n, n * l));
    for i in range(n):
      ii[ i, (i * l):((i+1) * l) ] = z;
    
    ww = self.weights[:n, :n];
    
    pp = np.dot(ww, self.p) + ii;
    
    #print pp.shape
    #print ww.shape
    #print df.shape
    #print self.state.shape
    
    self.p = (pp.T * df).T;
    
    # error
    ee = np.zeros(n);
    ee[self.outputNodes] = self.goal - self.output;
    
    #print ee.shape
    
    #update weights
    rr = self.eta * np.dot(ee, self.p);
    rr = rr.reshape((n, l))
    #print rr.shape   
    
    self.weights += rr;
    self.constrainWeights();

    

  
  

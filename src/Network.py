"""
Network Base Class

Setps up basic structure
for an artificial neuronal network
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np
from NetworkIOStreams import NetworkIOStream;

# Network classes

class Network:
  """
  Network base class for the simulation and learning of
  a recurrent network, discrete time
  """

  def __init__(self, nNodes, io = NetworkIOStream()):
    """Initialize network

    Arguments:
      nNodes (int): number of network nodes
      nInputs (int): number of input nodes
      outputNodes (array): indices of nodes representing outputs
    """

    self.nNodes = nNodes;
    self.io = io;

    self.nInputs = io.nInputs;
    #self.inputNodes = range(self.nInputs);

    self.nOutputs = io.nOutputs;
    #self.outputNodes = range(self.nInputs, self.nInputs + self.nOutputs);
    self.outputNodes = range(0, self.nOutputs);

    self.weights = 0.2 * (2 * np.random.rand(nNodes, nNodes + self.nInputs) - 1);
    self.state = np.random.rand(nNodes);
    self.input  = np.zeros(self.nInputs);
    self.output = np.zeros(self.nOutputs);
    self.goal   = np.zeros(self.nOutputs);


  def step(self):
    """Evaluate time step"""

    self.input = self.io.getInput();
    self.goal  = self.io.getOutput();

    # y_k = f(s_k(t))
    self.state = self.f(self.weights * np.hstack((self.state, self.input)));
    self.output = self.state[self.outputNodes];


  def constrainWeights(self):
    """Constrain no-recurrent weights to output layers"""
    self.weights[np.ix_(self.outputNodes, self.outputNodes)] = 0;


  def error(self):
    return 0.5 * np.sum(np.power(self.output - self.goal, 2));


  def update(self, delta_weights):
    self.weights += delta_weights;

  def f(self, x):
    return (1./(1. + np.exp(-x)));

  def currentState(self):
    return self.weights

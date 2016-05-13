"""
Real Time Recurrent Modulated Learning

Class implementing real time leanring in a network with modulatory connections

Reference
---------
  C. Kirst 2016, Real Time Recurrent Modulatory Learning, Notes

"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np
from Network import Network;
from NetworkIOStreams import NetworkIOStream;


class RTRMLNetwork(Network):
  """Real Time Recurrent Modulatory Learning Network"""

  def __init__(self, nNodes, io = NetworkIOStream(), eta = 0.1, gamma = 0.1, reset = None):
    """Initialize network

    Arguments:
      nNodes (int): number of network nodes
      io (NetworkIOStream): teaching sequence generator
      eta (float): learning rate for weights
      gamma (float): learning rate for modulation
      reset (int): reset the sensitivities after this number of steps (None = never reset)
    """

    Network.__init__(self, nNodes, io);
    self.eta = eta;
    self.gamma = gamma;
    self.resetNetwork();
    self.reset = reset;
    self.resetCounter = 0;


    n = self.nNodes;
    m = n + self.nInputs;

    # 0.5 is the average
    # so the mean is zero
    self.state = np.random.rand(n);
    self.weights = np.random.rand(n, m) - 0.5;

    self.a = np.random.rand(n, m) - 0.5;    #a[i,j]
    self.b = np.random.rand(n, n, m) - 0.5; #b[k,i,j]

    self.weights = self.a + self.g(np.tensordot(self.b, self.state, axes = (0,0)));

    self.constrainWeights();


  def resetNetwork(self):
    """Reset the network p's and q's to initial state"""

    n = self.nNodes;
    m = n + self.nInputs;

    #print n,m
    # p[k,i,j]
    self.p = np.zeros((n, n, m));
    # q[k,l,i,j]
    self.q = np.zeros((n, n, n, m));
    self.t = 0;


  # f(x) and g(x) are smoothing functions
  # for the output of the network.
  # so that everything remains normalized
  # What happens when things aren't normalized?
  def f(self, x):
    return (1./(1. + np.exp(-x)));

  def g(self, x):
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
    m = n + self.nInputs;

    # generate input and output
    self.input  = self.io.getInput();
    self.goal   = self.io.getOutput();

    # update weights and states
    z = np.hstack((self.state, self.input));

    g = self.g(np.tensordot(self.b, self.state, axes = (0,0)))
    w = self.a + g;
    y = self.f(np.dot(w,  z));

    # update sensitivities
    df = y * (1 - y);
    dg = g * (1 - g);

    # weight sensitivities
    ii = np.zeros((n, n, m));
    for i in range(n):
      ii[i, i, :] = z;

    ww = self.weights[:n, :n] + np.einsum('km,lkm,m->kl',dg[:,:n], self.b[:,:,:n], self.state);
    self.p = np.einsum('kl,lij->kij', ww, self.p) + ii;
    self.p = np.einsum('k,kij->kij', df, self.p);
    #print self.p.shape

    # modulatory sensitivities  b[k,m,i,j]
    ii = np.zeros((n, n, n, m));
    for i in range(n):
      ii[ i, :, i, :] = np.einsum('j,j,m->mj', dg[i,:], z, self.state);

    #ww = self.weights[:n, :n] + np.einsum('km,lkm,m->kl',dg[:,:n], self.b[:,:,:n], self.state);
    self.q = np.einsum('kl,lmij->kmij', ww, self.q) + ii;
    self.q = np.einsum('k,kmij->kmij', df, self.q)
    #print self.q.shape
    #print '------------'


    self.state = y;
    self.weights = w;
    #self.constrainWeights()
    self.output = y[self.outputNodes];


    # error signals / learning

    ee = np.zeros(n);
    ee[self.outputNodes] = self.goal -  self.output;
    #print ee.shape

    #update a's and b's
    # as the learning rate. It multiplies the derivative by the error.
    self.a += self.eta * np.tensordot(self.p, ee, axes= (0,0));
    self.b += self.gamma * np.tensordot(self.q, ee, axes = (0,0));

"""
Network Teaching Streams

Collection of IO streams of teaching sequence generators in time for e.g. RTRL.

So the network learns to predict the training sequence 
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np

# IO Streams for training

class NetworkIOStream:
  """Base Network IO stream for training, constant input for thresholding"""

  def __init__(self):
    self.nInputs = 1;
    self.nOutputs = 0;

  def getInput(self):
    return 1;

  def getOutput(self):
    return [];

from scipy.signal import convolve2d
class AutomatonIOStream(NetworkIOStream):
    @staticmethod
    def game_of_life_step(state):
        X = state.astype(bool)
        nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
        ret = (nbrs_count == 3) | (X & (nbrs_count == 2))
        return ret.astype(int)
    
    @staticmethod
    def init_random(w,h):
        start = np.zeros(w*h)
        start[:w*h*0.3]=1
        np.random.shuffle(start)
        then = start.reshape((w,h))
        return then
    
    def reinitialize(self):
        self.input = self.init_random(self.size,self.size)
#         print self.input
        self.output = self.stepper(self.input)
        self.t=0
    
    def __init__(self, 
                 size=10, 
                 interval=10,
                 stepper=game_of_life_step.__func__):
        
        self.size=size
        self.stepper=stepper
        self.nOutputs=size**2
        self.nInputs=size**2
        self.reinit_interval=interval
        self.reinitialize()
        print self.input
        print self.output
        
    def getInput(self):
        if self.t==(self.reinit_interval):
            self.reinitialize()
        else:
            self.input=self.output
            self.output=self.stepper(self.input)
#         print "Input: "
        self.t+=1
        return self.input.reshape((self.nInputs,))
        
    def getOutput(self):
        return self.output.reshape((self.nOutputs,))

class DelayedIOStream(NetworkIOStream):
  """Generates input output data for delayed Xor"""

  def __init__(self, delay = 2):
    self.nInputs = 1 + 1;
    self.nOutputs = 1;
    self.delay = delay;
    self.history = np.zeros((delay, 2));
    self.nextout = 0;
    self.output = np.array(0);


  def getInput(self):
    self.output = self.history[self.nextout,:1].copy();

    sequence = np.random.randint(0,2,2);
    sequence[-1] = 1; # constant input for thresholding

    self.history[self.nextout, :] = sequence;
    self.nextout = (self.nextout + 1) % self.delay

    return sequence;

  def getOutput(self):
    return self.output;




class XorIOStream(NetworkIOStream):
  """Generates input output data for delayed Xor"""

  def __init__(self, delay = 2):
    self.nInputs = 2 + 1;
    self.nOutputs = 1;
    self.delay = delay;
    self.history = np.zeros((delay, 3));
    self.nextout = 0;
    self.output = np.array(0);


  def getInput(self):

    self.output = np.array([np.logical_xor(self.history[self.nextout,0], self.history[self.nextout,1])], dtype = 'int');

    sequence = np.random.randint(0,2,3);
    sequence[-1] = 1; # constant input for thresholding

    self.history[self.nextout, :] = sequence;
    self.nextout = (self.nextout + 1) % self.delay

    return sequence;


  def getOutput(self):
    return self.output;



class ContextXorAndIOStream(NetworkIOStream):
  """Generates input output data for delayed Xor"""

  def __init__(self, delay = 2):
    self.nInputs = 3 + 1;
    self.nOutputs = 1;
    self.delay = delay;
    self.history = np.zeros((delay, 3));
    self.next = 0;
    self.output = np.array(0);


  def getInput(self):

    sequence = np.random.randint(0,2,4);
    sequence[-1] = 1; # constant input for thresholding

    self.history[self.next, :] = sequence[:3];
    self.next = (self.next + 1) % self.delay

    return sequence;


  def getOutput(self):
    s = self.history[self.next,:];
    if s[2]:
      return np.array([np.logical_xor(s[0], s[1])]).astype('float');
    else:
      return np.array([np.logical_and(s[0], s[1])]).astype('float');



class SequenceRecognitionIOStream(NetworkIOStream):
  """Generates input output data for sequence recognition"""

  def __init__(self, m = 2, state = -2, a = 0, b = 1, delay = 0):
    self.nInputs = m + 1;
    self.nOutputs = 1;
    self.m = m;
    self.a = a;
    self.b = b;
    self.delay = delay;
    self.state = -2;
    self.sequence = np.zeros(m);
    self.output = np.array(0);


  def getInput(self):
    sequence = np.zeros(self.nInputs);
    sequence[np.random.randint(0,self.m)] = 1;

    self.output = [0];
    if self.state == -2:
      if sequence[self.a] == 1:
        self.state = -1;
    elif self.state == -1:
      if sequence[self.b] == 1:
        self.state = 0;

    if self.state == self.delay:
        self.state = -2;
        self.output = [1];
    elif self.state >= 0: # integrate delay
        self.state += 1;
        #no inputs
        sequence = np.zeros(self.nInputs);

    sequence[-1] = 1; # constant input for thresholding
    #self.sequence = sequence;

    return sequence;


  def getOutput(self):
    return self.output;




class RatchetIOStream(NetworkIOStream):
  """Generates input output data of a perfect path integrator"""

  def __init__(self, nOrientations = 5, maxStep = 1, jumpProbability = 0.01, outputDelay = 0, network = None):
    self.nOrientations = nOrientations;
    self.nInputs = (maxStep + 1) + 1;
    self.nOutputs = nOrientations;
    #self.nOutputs = nOrientations - 1;

    self.maxStep = maxStep;

    self.position = 0;

    self.output = np.zeros(self.nOrientations);
    #self.output = np.zeros(self.nOrientations - 1);
    self.output[self.position] = 1;

    #self.input = np.zeros(self.nInputs);
    #self.input[self.position] = 1;

    self.outputDelay = outputDelay;
    self.jumpProbability = jumpProbability;
    self.jumpHistory = np.zeros(self.outputDelay + 1);
    self.delay = 0;
    self.network = network;


  def getInput(self):
    delay =  self.delay;

    # generate signal
    jump = np.random.rand(1) < self.jumpProbability;
    if jump:
      jump = np.random.randint(1, self.maxStep+1);
    else:
      jump = 0;

    sig = np.zeros(self.nInputs);
    #sig[self.maxStep + jump] = 1;
    #if jump < 0:
    #  sig[self.maxStep + jump] = 1;
    #elif jump > 0:
    #sig[jump-1] = 1;
    sig[jump] = 1;
    sig[-1] = 1; # constant input for thresholding

    #update output
    delta = self.jumpHistory[delay];

    if self.network is None: # normal integration
      self.position = (self.position + delta) % self.nOrientations;
      self.output = np.zeros(self.nOrientations);
      self.output[self.position] = 1;
      #self.output = self.output[:-1];
    else:
      if np.random.rand() < 0.1:
        out = self.network.output;
        self.network.resetNetwork();
      else:
        out = self.output;

      if delta != 0:
        maxpos = np.argmax(out);
        #self.position = (maxpos + delta) % self.nOrientations;
        #self.output = np.zeros(self.nOrientations);
        #self.output[self.position] = 1;
        xmax = out[maxpos];
        out = out * out;
        out[maxpos] = np.sqrt(xmax);
        self.output = np.roll(out, int(delta));
      else:
        #out = self.output;    # we keep output of currnet state and inly amplify peak
        maxpos = np.argmax(out);
        xmax = out[maxpos];
        out = out * out;
        out[maxpos] = np.sqrt(xmax);
        self.output = out;

    #update jump history
    self.jumpHistory[delay] = jump;

    #update delay
    self.delay = (delay + 1) % (self.outputDelay + 1);

    return sig;

  def getOutput(self):
    return self.output;


class PathIntegrationIOStream(NetworkIOStream):
  """Generates input output data of a perfect path integrator"""

  def __init__(self, nOrientations = 8, maxStep = 1, period = 1, switch = 0):
    self.nOrientations = nOrientations;
    #self.nInputs = (2 * maxStep + 1) + 1;
    self.nInputs = (2 * maxStep) + 1;
    #self.nInputs = (maxStep) + 1;
    self.nOutputs = nOrientations;

    self.maxStep = maxStep;

    self.position = 0;

    self.output = np.zeros(self.nOrientations);
    self.output[self.position] = 1;

    #self.input = np.zeros(self.nInputs);
    #self.input[self.position] = 1;

    self.switch = switch;

    self.period = period;
    self.nextStep = period;


  def getInput(self):
    self.nextStep -= 1;

    if self.nextStep > 0:

      #no sustained input:
      sig = np.zeros(self.nInputs);
      sig[-1] = 1;

    else: # period is renewed
      self.nextStep = self.period;

      delta = np.random.randint(-self.maxStep, self.maxStep+1);
      #delta = np.random.randint(0, self.maxStep+1)
      self.position = (self.position + delta) % self.nOrientations;


      sig = np.zeros(self.nInputs);
      #sig[self.maxStep + delta] = 1;
      if delta < 0:
        sig[self.maxStep + delta] = 1;
      elif delta > 0:
        sig[delta-1] = 1;

      sig[-1] = 1; # constant input for thresholding


    if self.nextStep == self.period - self.switch:
      self.output = np.zeros(self.nOrientations);
      self.output[self.position] = 1;

    return sig;

  def getOutput(self):
    return self.output;






import reber;

class ReberIOStream(NetworkIOStream):
  """Generates input output data of a perfect path integrator"""


  def __init__(self, network = None):
    self.nInputs = 7;
    self.nOutputs = 7;
    self.generate();

  def generate(self):
    self.input_str = reber.make_reber();
    self.input_seq = reber.str_to_vec(self.input_str);
    self.ouput_seq = reber.str_to_next(self.input_str);
    self.step = 0;
    self.max_step = len(self.input_str);

  def getInput(self):
    self.step += 1;
    if self.step == self.max_step:
      self.generate();
      return None; # signals reset in LSTM

    else:
      return self.input_seq[self.step,:];


  def getOutput(self):
    return self.ouput_seq[self.step,:];

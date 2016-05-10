"""
Network Viewer

Visualizes the dynamics of a network and assoicated parameters during learning
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


#import sys, os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

#import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree

class NetworkViewer(QtGui.QWidget):
  """Visualize Dynamics of Network"""
  
  def __init__(self, network, timer):
    """Constructor"""
    QtGui.QWidget.__init__(self);
  
    self.network = network;

    self.colors = {"output" : QtGui.QColor(0, 0, 255), 
                   "error"  : QtGui.QColor(255, 0, 0), 
                   };
    self.timer = timer;
    
    self.setupGUI();
        
        
  def setupGUI(self):
    """Setup GUI"""
    self.layout = QtGui.QVBoxLayout()
    self.layout.setContentsMargins(0,0,0,0)        
    self.setLayout(self.layout)
    
    self.splitter = QtGui.QSplitter()
    self.splitter.setOrientation(QtCore.Qt.Vertical)
    self.splitter.setSizes([int(self.height()*0.5), int(self.height()*0.5)]);
    self.layout.addWidget(self.splitter)

    self.splitter2 = QtGui.QSplitter()
    self.splitter2.setOrientation(QtCore.Qt.Horizontal)
    #self.splitter2.setSizes([int(self.width()*0.5), int(self.width()*0.5)]);
    self.splitter.addWidget(self.splitter2);
    
    self.splitter3 = QtGui.QSplitter()
    self.splitter3.setOrientation(QtCore.Qt.Horizontal)
    #self.splitter2.setSizes([int(self.width()*0.5), int(self.width()*0.5)]);
    self.splitter.addWidget(self.splitter3);
    
    
    # various matrix like plots: state, goal, weights, p
    self.nStates = 20; #number of states in the past to remember
    self.matrixdata = [np.zeros((self.network.nInputs, self.nStates)),
                       np.zeros((self.network.nOutputs, self.nStates)), 
                       np.zeros((self.network.nOutputs, self.nStates)),
                       np.zeros((self.network.nNodes,   self.nStates))];
                       #np.zeros(self.network.weights.shape)];
                       #np.zeros(self.network.a.shape)
                       #np.zeros(self.network.b.shape)];
                       
    self.images = [];
    for j in range(len(self.matrixdata)):
      l = pg.GraphicsLayoutWidget()
      self.splitter2.addWidget(l);
      v = pg.ViewBox();
      l.addItem(v, 0, 1);
      i = pg.ImageItem(self.matrixdata[j]);
      v.addItem(i);
      self.images.append(i);
    
    for i in [0,1,2,3]:
      self.images[i].setLevels([0,1]);
    
    #output and error
    self.plotlayout = pg.GraphicsLayoutWidget();
    self.splitter3.addWidget(self.plotlayout);
    
    self.plot = [];
    for i in range(2):
      self.plot.append(self.plotlayout.addPlot());
      self.plot[i].setYRange(0, 1, padding=0);
    
    self.plot[1].setYRange(0, 0.5, padding=0)      

      
    self.plotlength = 2000;
    self.output = np.zeros((self.network.nOutputs, self.plotlength));
    #self.goal   = np.zeros((self.network.nOutputs, self.plotlength));
    self.errorlength = 2000;
    self.error  = np.zeros(self.errorlength);     
      
    self.curves = []
    for i in range(self.network.nOutputs):
      c = self.plot[0].plot(self.output[i,:], pen = (i, self.network.nOutputs));
      #c.setPos(0,0*i*6);
      self.curves.append(c);      
    
    c = self.plot[1].plot(self.error, pen = (2,3));
    self.curves.append(c);
    
    
    # parameter controls
    self.steps = 0;
    
    params = [
        {'name': 'Controls', 'type': 'group', 'children': [
            {'name': 'Simulate', 'type': 'bool', 'value': True, 'tip': "Run the network simulation"},
            {'name': 'Plot', 'type': 'bool', 'value': True, 'tip': "Check to plot network evolution"},
            {'name': 'Plot Interval', 'type': 'int', 'value': 10, 'tip': "Step between plot updates"},
            {'name': 'Timer', 'type': 'int', 'value': 10, 'tip': "Pause between plot is updated"},
        ]}
        ,
        {'name': 'Network Parameter', 'type': 'group', 'children': [
            {'name': 'Eta', 'type': 'float', 'value': self.network.eta, 'tip': "Learning rate"}#,
            #{'name': 'Gamma', 'type': 'float', 'value': self.network.gamma, 'tip': "Learning rate"},
        ]}
        ,
        {'name': 'Status', 'type': 'group', 'children': [
            {'name': 'Steps', 'type': 'int', 'value': self.steps, 'tip': "Actual iteration step", 'readonly': True}
        ]}
      ];

    self.parameter = Parameter.create(name = 'Parameter', type = 'group', children = params);
    
    print self.parameter
    print self.parameter.children()
    
    self.parameter.sigTreeStateChanged.connect(self.updateParameter);   
    
    ## Create two ParameterTree widgets, both accessing the same data
    t = ParameterTree();
    t.setParameters(self.parameter, showTop=False)
    t.setWindowTitle('Parameter');
    self.splitter3.addWidget(t);
    
    # draw network
    self.nsteps = 100;    
    self.updateView();
    
       
  def updateParameter(self, param, changes):
    for param, change, data in changes:
      prt = False;
      if param.name() == 'Eta':
        self.network.eta = data;
        prt = True;
      if param.name() == 'Gamma':
        self.network.gamma = data;
        prt = True;
      elif param.name() == 'Timer':
        self.timer.setInterval(data);
        prt = True;
        
      if prt:
        path = self.parameter.childPath(param);
        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()
        print('  parameter: %s'% childName)
        print('  change:    %s'% change)
        print('  data:      %s'% str(data))
        print('  ----------')
      
        
  def updateView(self):
    """Update plots in viewer"""
    
    if self.parameter['Controls', 'Simulate']:
      
      pl = self.parameter['Controls', 'Plot'];
      ns = self.parameter['Controls', 'Plot Interval'];
      
      if pl:
        for i in range(len(self.matrixdata)):
          self.matrixdata[i][:, :-ns]   = self.matrixdata[i][:,ns:];
          
        self.output[:, :-ns] = self.output[:, ns:];
        self.error[:-1] = self.error[1:];
    
      for i in range(ns,0,-1):
          self.network.step();
          
          if pl:
            self.output[:,-i]    = self.network.output;
            self.matrixdata[0][:,-i] = self.network.input;
            self.matrixdata[1][:,-i] = self.network.goal;
            self.matrixdata[2][:,-i] = self.network.output;
            self.matrixdata[3][:,-i] = self.network.state;
            
            
      self.steps += ns;
      self.parameter['Status', 'Steps'] = self.steps;
      
      if pl:
        
        #s = self.network.state.copy();
        #s = s.reshape(s.shape[0], 1);
        
        for i in range(len(self.matrixdata)):
          self.images[i].setImage(self.matrixdata[i]);
    
        #self.images[-1].setImage(self.network.weights.T);
        #self.images[-1].setImage(self.network.p.T);
        
        # update actual state
        #self.curves[0].setData(self.network.output);
        #self.curves[1].setData(self.network.goal.astype('float'));
        
        # update history
        for i in range(self.network.nOutputs):
          self.curves[i].setData(self.output[i,:]);
        
      #keep updating error as its cheap
      self.error[-1:] = self.network.error();
      self.curves[-1].setData(self.error);
      
    
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 00:30:44 2016

@author: ckirst
"""


import NetworkViewer2 as netview;
import RTRML as network;
import NetworkIOStreams as nio;

import sys          
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import matplotlib.pyplot as plt;

import numpy as np

reload(netview); reload(network);

net = network.RTRMLNetwork;

rtrlnet = net(nNodes = 5, io = nio.XorIOStream(delay = 2), eta = 0.9, gamma = 1);
rtrlnet.a  = 0 * np.ones_like(rtrlnet.weights);

#rtrlnet.b = np.zeros_like(rtrlnet.b);
#rtrlnet = net(nNodes = 20, io = nio.PathIntegrationIOStream(nOrientations = 4, period = 20, switch = 1), eta = 0.1);

#rtrlnet = net(nNodes = 6, io = nio.SequenceRecognitionIOStream(m = 2, delay = 30), eta = 0.1, reset = 100);

#io = nio.RatchetIOStream(nOrientations = 4, jumpProbability = 0.05, outputDelay = 0);
#rtrlnet = net(nNodes = 20, io = io, eta = 0.1, gamma = 0,  reset = None);
#io.network = rtrlnet;



pg.mkQApp()

reload(netview)

t = QtCore.QTimer()

win = netview.NetworkViewer(rtrlnet, timer = t);
win.setWindowTitle("Real Time Recurrent Learning Simulator");
win.show()
win.resize(1100,700)

#if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#    QtGui.QApplication.instance().exec_()
    
def updateRandom():
  win.updateView();

t.timeout.connect(updateRandom)
t.start(10)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        
        
        





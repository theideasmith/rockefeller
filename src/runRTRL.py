# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 00:30:44 2016

@author: ckirst
"""


import NetworkViewer as netview;
import RTRL as network;
import NetworkIOStreams as nio;

import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import matplotlib.pyplot as plt;

import numpy as np
from NetworkExamples import *
reload(netview); reload(network);


pg.mkQApp()

reload(netview)

t = QtCore.QTimer()

win = netview.NetworkViewer(automaton, timer = t);
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





# if False:
#   g = [];
#   o = [];
#   for s in range(100000):
#     rtrlnet.step();
#     #print rtrlnet.state
#     g.append(rtrlnet.goal);
#     o.append(rtrlnet.output[0]);
#
#   plt.figure(14); plt.clf()
#   plt.scatter(np.linspace(0, len(o), len(o)), np.array(g) - np.array(o));
#   #plt.plot(o);
#
#
# if False:
#   import RTRLNetwork as net;
#   import matplotlib.pyplot as plt
#   import numpy as np
#
#   g= net.RatchetIOStream(nOrientations = 10, jumpProbability = 0.5, outputDelay = 0);
#   inp = [];
#   out = [];
#   for i in range(100):
#     inp.append(g.getInput()[1]);
#     out.append(np.argmax(g.getOutput()));
#
#   fig = plt.figure(1); fig.clf;
#   plt.plot(inp)
#   plt.plot(out)

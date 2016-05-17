from Cluster import *
from NetworkExamples import *
from pylab import *
r = RunNetwork(automaton)
r.collect(log=True)

print r.getSolutions()


from scipy.integrate import *
from numpy import *
import matplotlib.pyplot as plt
K = 0.2
M = 2.3
def k(x): return 1.0-(x/M)
def logistic(x,t): 
	"""
	I just made the very interesting connection that 
	the sigmoid function is a solution to the 
	logistic growth differential
	equaiton. 

	This enables me to think of the sigmoid 
	function in terms of what logistic growth
	means. 

	It means that as values cross the inflection
	point of the derivative, f(x), become smaller
	smaller. This means larger and larger values
	make smaller changes. 
	"""
	# k(x)*K*x+r
	res = x*0.2+random.random()*10
	print res
	return res


time = arange(0,100,0.1)
start = array([0.1])

state = odeint(logistic, start, time, mxstep=50000000)
plt.plot(time, state[:,0]) 
plt.show()

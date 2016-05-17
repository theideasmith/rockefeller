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










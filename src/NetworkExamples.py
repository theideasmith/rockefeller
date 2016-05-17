import NetworkIOStreams as nio
import RTRL as network
net = network.RTRLNetwork;                                                                                            
                                                                                                                      
xor_stream = net(nNodes = 3, io = nio.XorIOStream(delay = 2), eta = 0.5);                                                

path_integration = net(nNodes = 20, io = nio.PathIntegrationIOStream(nOrientations = 4, period = 20, switch = 1), eta = 0.1); 
                                                                                                                      
sequence_recognition = net(nNodes = 6, io = nio.SequenceRecognitionIOStream(m = 2, delay = 30), eta = 0.1, reset = 100);          
ratchet_io = nio.RatchetIOStream(nOrientations = 4, jumpProbability = 0.05, outputDelay = 0);                               
ratchet_io_stream = net(nNodes = 20, io = ratchet_io, eta = 0.1, reset = None);                                            
ratchet_io.network = ratchet_io_stream                                                                                               

automaton = net(50, nio.AutomatonIOStream(size=4))


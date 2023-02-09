import numpy as np
import random
import matplotlib.pyplot as plt

#=================================================================
activation = "linear"
alpha = 0.6
lr = 0.1 ## Learning Rate

struct = [2,2,1]
#=================================================================


md = 5 #max decimal
def dm(x): return round(x,md) 


## BINARY
def Activate(function,x,alpha): ## GENIAL ----------

    ## TANH
    if(function=="tanh"): return np.tanh(x*1)
    ## TANH Positive
    if(function=="tanhp"): 
        p = np.tanh(x*1)
        if(p>0): return p
        if(p<=0): return 0
    ## SIGMOID
    if(function=="sigmoid"): return 1 / (1 + np.exp((-x*1)))
    ## BINARY
    if(function=="binary"):  ## GENIAL ----------
        if(x>0): return 1
        else: return 0
    ## RELU
    if(function=="relu"):
        if(x>a): return x
        else: return 0
    ## LINEAR
    if(function=="linear"): return x


class Neuron:
    def __init__(self, weights):
        self.weight = []
        self.neuronv = 0 ## Neuron Value
        self.neuronvN = 0 ## Neuron Value NO ACTIVATION 
        self.bias = dm(random.uniform(-1, 1))
        for i in range(0, weights):
            self.weight.append(dm(random.uniform(-1, 1)))

class Layer:
    def __init__(self, neurons, weights):
        self.neuronl = []
        for i in range(0, neurons):
            self.neuronl.append(Neuron(weights))
            self.neuronl[i].__init__(weights)

class Network:
    def __init__(self, Structure):
        self.layers = []
        self.structure = []
        structure = Structure

        self.histfullerror = []
        self.histNeuronResultD = []
        
        for i in range(0, len(Structure)-1):
            self.layers.append(Layer(structure[i+1], structure[i]))

#===========================================================
    def state(self): ## PRINT NETWORK STATE AND VALUES
        for l in range(0, len(self.layers)):
            print("---------------------------------")
            for n in range(len(self.layers[l].neuronl)):
                print("Layer: ", l ," - Neuron: ", n)
                print("Weights: ", self.layers[l].neuronl[n].weight)
                print("Bias: ", self.layers[l].neuronl[n].bias)
                print("Value: ", self.layers[l].neuronl[n].neuronv)
#===============================================================
    def CopyWeights(self, NetworkIn, NetworkOut): 
        #each Layer
        for l in range(0, len(NetworkIn.layers)):
            #each Neuron
            for n in range(len(NetworkIn.layers[l].neuronl)):
                #each Weight
                for w in range(len(NetworkIn.layers[l].neuronl[n].weight)):
                    weightA = NetworkIn.layers[l].neuronl[n].weight[w]
                    NetworkOut.layers[l].neuronl[n].weight[w] = weightA
                biasA = NetworkIn.layers[l].neuronl[n].bias
                NetworkOut.layers[l].neuronl[n].bias = biasA
#===============================================================
    def predict(self, Input):
        global activation
        global alpha
        #for each Neuron (and Input)
        for n in range(len(self.layers[0].neuronl)):
            NeuronResult = 0
            #for each data in Input
            for i in range(len(Input)):
                NeuronResult = NeuronResult + (Input[i] * self.layers[0].neuronl[n].weight[i])
            #print(NeuronResult)
            NeuronResult += self.layers[0].neuronl[n].bias
            #print(NeuronResult)
            #NeuronResult = Activation(fa,NeuronResult,alpha)
            #print(NeuronResult)
            self.layers[0].neuronl[n].neuronvN =  NeuronResult
            self.layers[0].neuronl[n].neuronv =  Activate(activation,NeuronResult,alpha)
        #each Layer
        for l in range(1, len(self.layers)):
            #each Neuron
            for n in range(len(self.layers[l].neuronl)):
                NeuronResult = 0
                #each Neuron in LayerBefore
                for nlb in range(len(self.layers[l-1].neuronl)):
                    #value of neuron in the layer before
                    vnlb = self.layers[l-1].neuronl[nlb].neuronv
                    #weight of neuron in actual layer
                    wn = self.layers[l].neuronl[n].weight[nlb]
                    #print(NeuronResult,vnlb,wn)
                    NeuronResult = NeuronResult + (vnlb * wn)
                NeuronResult += self.layers[l].neuronl[n].bias
                self.layers[l].neuronl[n].neuronv = Activate(activation,NeuronResult,alpha)
                self.layers[l].neuronl[n].neuronvN = NeuronResult
#===============================================================
    def getOutput(self): ## GET THE LAST LAYER VALUES
        outneurons = []
        for n in range(len(self.layers[len(self.layers)-1].neuronl)):
            outneurons.append(self.layers[len(self.layers)-1].neuronl[n].neuronv)
        return outneurons
        #return 1
#===============================================================
    def PEPITAinit(self):
        self.NetworkErrorTest = Network(struct)
        
    def PEPITAtrain(self, Input, Output):
        global lr
        global activation
        global alpha
        NeuronResultD=0
        
        #print("< Step 1 > Normal Foward Pass")
        #print("Training Set: ",Input,Output)
        self.predict(Input)
        lastlayer = len(self.layers)-1
        #print("Prediction:",self.layers[lastlayer].neuronl[0].neuronv)
#=====================================================================
# GET TOTAL ERROR 
#=====================================================================
        self.errortotal = 0
        #print("< Step 2 > Calculate Error between prediction and target")
        
        erroroutputs = [] #an array of each error in each output neuron
        #for each neuron in the last layer
        for n in range(len(self.layers[lastlayer].neuronl)):
            errorcalculate = Output[n] - self.layers[lastlayer].neuronl[n].neuronv            
            erroroutputs.append(errorcalculate)
            self.errortotal += errorcalculate 
            #print("> Target Output:",Output[n],"Network Output:",self.layers[lastlayer].neuronl[n].neuronv)
        if(self.errortotal<-100):self.errortotal=-100
        if(self.errortotal>100):self.errortotal=100
        #print("SUM total error:",self.errortotal)
        self.histfullerror.append(self.errortotal)
#===============================================================
        #print("< Step 3 > Put the error in the Input")
        InputE = [] #COMBINATION OF INPUT AND ERROR
        for i in range(len(Input)):
            #print("Input[",i,"]:", Input[i], " + ErrorTotal:", self.errortotal)
            F=1 #?
            IE=((Input[i]) + (self.errortotal*F))
            InputE.append(IE)
        #print("Input", Input)
        #print("InputE", InputE)
#=================================================================
        #print("< Step 4 > Fowardpass Network with Error in the Input")
        #Copy Weight of the network to the other network, NetworkErrorTest
        self.CopyWeights(self, self.NetworkErrorTest)
        #predict the network with error
        self.NetworkErrorTest.predict(InputE)
#=================================================================
        #print("< Step 5 > Compare neurons in both Network (with and without error)
        for l in range(0, len(self.layers)):
            #for each Neuron
            for n in range(len(self.layers[l].neuronl)):
                ## NeuronResultD is the difference between the neurons of the network without error and the network with the propagated error
                NeuronResultD = self.layers[l].neuronl[n].neuronvN - self.NetworkErrorTest.layers[l].neuronl[n].neuronvN
                if(NeuronResultD>100):NeuronResultD=100
                if(NeuronResultD<-100):NeuronResultD=-100
                self.histNeuronResultD.append(NeuronResultD)
#=================================================================
                #print("< Step 6 > Change Weights")
                #each Weight
                for w in range(len(self.layers[l].neuronl[n].weight)):
                    if(l==0): #If first hidden layer, Compare with inputs
                        self.layers[l].neuronl[n].weight[w] += (lr * (NeuronResultD) * (InputE[w]))
                    else: #Compare with layer-1
                        ## Backpropagation
                        #self.layers[l].neuronl[n].weight[w] += (lr * self.errortotal * self.NetworkErrorTest.layers[l-1].neuronl[w].neuronv)
                        
                        # or
                        
                        ## PEPITA
                        self.layers[l].neuronl[n].weight[w] += (lr * (NeuronResultD) * (self.NetworkErrorTest.layers[l-1].neuronl[w].neuronv))
                    if(self.layers[l].neuronl[n].weight[w]<-100):self.layers[l].neuronl[n].weight[w]=-100
                    if(self.layers[l].neuronl[n].weight[w]>100):self.layers[l].neuronl[n].weight[w]=100
                #change the bias
                if(l<lastlayer): # if not last layer, Compare with NeuronResultD
                    self.layers[l].neuronl[n].bias += lr * NeuronResultD
                else: # Compare with self.errortotal
                    self.layers[l].neuronl[n].bias += lr * self.errortotal
                if(self.layers[l].neuronl[n].bias<-100):self.layers[l].neuronl[n].bias=-100
                if(self.layers[l].neuronl[n].bias>100):self.layers[l].neuronl[n].bias=100
#===============================================================
#Code
#===============================================================
#Training Set
TrainSetIn =([[1, 1],
             [1, 0],
             [0, 1],
             [0, 0]])

TrainSetOut =([[0],
             [1],
             [1],
             [0]])
# ========================================================
epocs = 20
repeatset = 5

NN = Network(struct)
NN.PEPITAinit()
# ========================================================
for i in range(epocs):
    print("EPOC: ", i)
    for its in range(len(TrainSetIn)):
        for repeats in range(repeatset):
            NN.PEPITAtrain(TrainSetIn[its], TrainSetOut[its])
# ====================================================================================
## DEBUG
for its in range(len(TrainSetIn)):
    print("=====================================")
    print("Example: ", TrainSetIn[its])
    NN.predict(TrainSetIn[its])
    print("RESULT: ", NN.getOutput())
NN.state()

plt.plot(NN.histfullerror, label = "Full error")
plt.plot(NN.histNeuronResultD, label = "Difference Between Neurons")

plt.legend()
plt.title('Error History')
plt.show()
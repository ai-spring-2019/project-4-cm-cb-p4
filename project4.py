"""
------------------------------------------------
Names:      Chad Morse and Chris Browne
Date:       5/1/19
Assignment: Project 4
Objective:  Implement forward propagating neural network that learns
            through back propagation

            in main call NeuralNetwork([3,4,3]) to create neural network
            with layers of 3,4,and 3 nodes respectively.
            call nn.backPropegateLearning(data) to train the network on the training data
            call nn.forwardPropegate(example) to retrieve predicted output class
            call crossValidation2(data) to test different network structures with a
            single hidden layer of 1-10 nodes.  Will output the cross-validated accuracy of
            each structure of [input, i, output] for i in range(1,10)
------------------------------------------------
"""

import csv, sys, random, math, copy
from statistics import mean
import pandas as pd

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)
    for pair in pairs:
        nn.forward_propagate(pair[0])
        class_prediction = nn.predict_class()
        if class_prediction != pair[1]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network
################################################################################

class NodeConnection:
    def __init__(self, first, second, weight):
        self.i = first
        self.j = second
        self.weight = weight

    def __repr__(self):
        return "[connection between node({},{}) and node({},{}) with weight: {}]".format(
        self.i.layerNum, self.i.layerIndex, self.j.layerNum, self.j.layerIndex, self.weight)

class Node:
    def __init__(self, layerNum, layerIndex):
        self.inj = 0
        self.ai = 0
        self.error = 0
        self.bias = random.random()
        self.layerIndex = layerIndex
        self.layerNum = layerNum
        self.inPaths = []
        self.outPaths = []

    def __str__(self):
        return "NODE"+str(self.layerIndex)

    def __repr__(self):
        return "NODE"+str(self.layerIndex)

    def getInWeights(self):
        inWeights = []
        for connection in self.inPaths:
            inWeights.append(connection.weight)
        return inWeights

    def getInActivations(self):
        inActivations = []
        for connection in self.inPaths:
            inActivations.append(connection.i.ai)
        return inActivations

    def getOutWeights(self):
        outWeights = []
        for connection in self.outPaths:
            outWeights.append(connection.weight)
        return outWeights

    def getOutErrors(self):
        outErrors = []
        for connection in self.outPaths:
            outErrors.append(connection.j.error)
        return outErrors

    def fixWeightsBias(self, learningRate):
        for connection in self.inPaths:
            connection.weight = connection.weight + (learningRate * connection.i.ai * self.error)
        #also fix the bias
        self.bias = self.bias + (learningRate * 1 * self.error)

    def updateInJandAI(self, activationFn):
        """dot of weights and inputs-- for this to work sthe previous layer's ai's must be set"""
        inj = 0
        for connection in self.inPaths:
            inj += (connection.i.ai * connection.weight)
        inj += self.bias
        self.inj = inj
        self.ai = activationFunction(inj, activationFn)

def nodeLayerConnect(firstNodes, nextNodes):
    for node in firstNodes:
        node.outPaths = []
    for node in nextNodes:
        node.inPaths = []#^both these loops shouldnt be necessary but just incase
    for i in firstNodes:
        for j in nextNodes:
            #for every path i to j
            newConnection = NodeConnection(i,j, random.random())
            i.outPaths.append(newConnection)
            j.inPaths.append(newConnection)
    return

class InputNode(Node):
    def __init__(self, layerNum, layerIndex):
        super().__init__(layerNum, layerIndex)

    def setInput(self, val):
        self.ai = val

class OutputNode(Node):
    def __init__(self, layerNum, layerIndex):
        super().__init__(layerNum, layerIndex)


class NeuralNetwork:
    def __init__(self, nodeNumberList, activationFunction = 'logistic'):
        assert(len(nodeNumberList) > 1) #must have at least input layer and output
        hiddenLayers = []
        inputNodes = [InputNode(0, i) for i in range(nodeNumberList[0])]
        outputNodes = [OutputNode(len(nodeNumberList) - 1, i) for i in range(nodeNumberList[-1])]
        if len(nodeNumberList) == 2:
            #no hidden layers
            nodeLayerConnect(inputNodes,outputNodes)
            self.inputNodes = inputNodes
            self.hiddenLayers = []
            self.outputNodes = outputNodes
            self.learningRate = .95
            self.activationFn = activationFunction
            self.lastForwardPropegation = None
            return
        for layer in range(len(nodeNumberList) - 2):
            hiddenLayers.append([Node(layer + 1,index) for index in range(nodeNumberList[layer+1])])
        #connect input nodes and first hidden layer
        nodeLayerConnect(inputNodes, hiddenLayers[0])
        #connect hidden layers
        for i in range(len(hiddenLayers) - 1):
            nodeLayerConnect(hiddenLayers[i], hiddenLayers[i+1])
        #connect last hidden layer to output nodes
        nodeLayerConnect(hiddenLayers[-1], outputNodes)
        self.inputNodes = inputNodes
        self.hiddenLayers = hiddenLayers
        self.outputNodes = outputNodes
        self.learningRate = 0.95
        self.activationFn = activationFunction
        self.lastForwardPropegation = None
        return

    def gPrime(self, x):
        return activationFunction(x, self.activationFn)*(1-activationFunction(x, self.activationFn))

    def forwardPropegate(self, input):
        #load input into input layer
        assert(len(input) == len(self.inputNodes)) #error check
        for i in range(len(self.inputNodes)):
            self.inputNodes[i].setInput(input[i])

        #continue down layers
        for layer in self.hiddenLayers + [self.outputNodes]:
            for node in layer:
                node.updateInJandAI(self.activationFn)

        #now collect ai of each output node into a list
        output = []
        for node in self.outputNodes:
            output.append(node.ai)
        self.lastForwardPropegation = output
        return output

    def backPropegateLearning(self, inputs, iterations = 1000): #should check iteration number
        # Weights are initialized when connection object is made
        for _ in range(iterations):
            for inp in inputs:
                for i in range(len(self.inputNodes)):
                    self.inputNodes[i].setInput(inp[0][i])
                    # ai ←xi

                for layer in self.hiddenLayers + [self.outputNodes]:
                    for node in layer:
                        node.updateInJandAI(self.activationFn)
                        # inj← sum(wi,j*ai)
                        # aj ←g(inj)

                for j in range(len(self.outputNodes)):
                    self.outputNodes[j].error = self.gPrime(self.outputNodes[j].inj) * (inp[1][j] - self.outputNodes[j].ai)
                    # Δ[j]←g′(inj) × (yj − aj)

                #self.hiddenLayers.reverse() ** used reverse instead, now dont have to worry about fixing the reverse after
                for reverseLayer in list(reversed(self.hiddenLayers)) + [self.inputNodes]:
                    for revNode in reverseLayer:
                        weights = revNode.getOutWeights()
                        errors = revNode.getOutErrors()
                        dotProd = dot_product(weights, errors)
                        revNode.error = self.gPrime(revNode.inj) * dotProd
                        # Δ[i] ← g′(ini) sum(wi,j Δ[j])

                for connectedLayer in self.hiddenLayers + [self.outputNodes]:
                    for node in connectedLayer:
                        node.fixWeightsBias(self.learningRate) #updated this occur from destination node rather than origin
                        # wi,j←wi,j + α × ai × Δ[j]

            # Change if we go until accuracy level is met
            self.learningRate = self.learningRate - (self.learningRate * (1 / iterations))

        return self

    def predict_class(self):
        predClass = []
        for i in self.lastForwardPropegation:
            predClass.append(int(round(i)))
        return predClass

    def forward_propagate(self, input):
        return self.forwardPropegate(input)

    def __repr__(self):
        return str([self.inputNodes] + self.hiddenLayers + [self.outputNodes])

################################################################################
### Activation Functions
################################################################################

def activationFunction(x, name):
    if name == 'logistic':
        return logistic(x)
    elif name == 'linear':
        return linear(x)
    elif name == 'tanh':
        return tanh(x)
    elif name == 'relu':
        return relu(x)
    elif name == 'leakyRelu':
        return leakyRelu(x)
    elif name == 'maxOut':
        return maxOut(x)
    else:
        print('INCORRECT ACTIVATION FUNCTION NAME')
        print('{} is not valid'.format(name))
        quit()

def linear(x):
    return x

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def tanh(x):
    """Tanh function"""
    try:
        numerator = 1 - math.e ** (-2 * x)
        denominator = 1 + math.e ** (-2 * x)
    except OverflowError:
        return 0.0
    return numerator / denominator

def relu(x):
    #https://isaacchanghau.github.io/post/activation_functions/
    return max(0,x)

def leakyRelu(x):
    pass

def maxOut(x):
    pass

def softmax(z):
    pass

################################################################################
### Cross Validation
################################################################################

def learner(size, examples):
    nnList = [len(examples[0][0])]
    for _ in range(size):
        nnList.append(random.randint(5,10))     # WHAT SHOULD THIS REALLY BE
    nnList.append(len(examples[0][1]))
    nn = NeuralNetwork(nnList)
    return nn.backPropegateLearning(examples)

def errorRate(hypothesis, examples):
    correct = 0
    count = 0
    for example in examples:
        fp = hypothesis.forwardPropegate(example[0])
        predicted = []
        for i in range(len(fp)):
            predicted.append(round(fp[i]))
        if predicted == example[1]:
            correct += 1
        count += 1
    return correct / count

def partition(examples, fold):
    validation = examples[fold]
    training = []
    for i in range(len(examples)):
        if i != fold:
            training += examples[i]
    return training, validation

def chunkList(l, k):
    chunks = []
    for i in range(0, len(l), k):
        chunks.append(l[i:i+k]) #check if works when len(l) not evenly divisible by K
    return chunks

def printEpoch(count, errorT, errorV):
    print("--------------------------------------------------------")
    print("Epoch Number: {}".format(count))
    print()
    print("Training Data Error:")
    print(errorT)
    print("Validation Data Error:")
    print(errorV)
    print()


def crossValidationWrapper(k, examples):
    errorTraining = []
    errorValidation = []
    size = 1
    exampleChunks = chunkList(examples, k)
    while True:
        crossValResults = crossValidation(size, k, exampleChunks)
        errorTraining.append(crossValResults[0])
        errorValidation.append(crossValResults[1])
        if min(errorTraining) < 0.1:     # Define converged here
            bestSize = errorValidation.index(min(errorValidation)) + 1
            return learner(bestSize, examples) #                                            **********
        printEpoch(size, errorTraining, errorValidation)
        size += 1

def crossValidation(size, k, exampleChunks):
    avgErrorTraining = 0
    avgErrorValidation = 0

    for fold in range(k):
        trainingSet, validationSet = partition(exampleChunks, fold)
        hypothesis = learner(size, trainingSet)#                                            ***********
        avgErrorTraining += errorRate(hypothesis, trainingSet)
        avgErrorValidation += errorRate(hypothesis, validationSet)
    return avgErrorTraining / k, avgErrorValidation / k


def confusionMatrix(nn, pairs):
    """creates confusionMatrix for accuracy check on nn using pairs"""
    numTargets = len(pairs[0][1])
    matrix = pd.DataFrame(index = ["predicted:{}".format(i) for i in range(numTargets)],
                          columns = ["target:{}".format(i) for i in range(numTargets)]).fillna(0)

    for example in pairs:
        pred = nn.forwardPropegate(example[0])
        predicted = "predicted:{}".format(round(pred.index(max(pred))))
        correct = "target:{}".format((example[1]).index(1))
        matrix.loc[predicted, correct] += 1
    print(matrix)

def binaryMatrix(nn,pairs):
    matrix = pd.DataFrame(index = ["predicted:{}".format(i) for i in [0,1]],
                          columns = ["target:{}".format(i) for i in [0,1]]).fillna(0)
    for example in pairs:
        pred = nn.forwardPropegate(example[0])
        predicted = "predicted:{}".format(round(pred[0]))
        correct = "target:{}".format(round(example[1][0]))
        matrix.loc[predicted, correct] += 1
    print(matrix)



def randomChunks(examples, chunkNumber=5):
    """breaks examples into chunkNumber random sets of approx same size and returns list of chunks"""
    shuffled = copy.deepcopy(examples)
    random.shuffle(shuffled) #want to make sure chunks fully random
    #want 5 approx equal sets
    chunkSize = int(len(shuffled)/chunkNumber)
    chunkList = []
    for i in range(0,len(shuffled),chunkSize):
        chunkList.append(shuffled[i:i+chunkSize])
    leftover = []
    while len(chunkList) > chunkNumber:
        leftover += chunkList.pop() #take remainders
    for i, ex in enumerate(leftover):
        chunkList[i%5].append(ex) #distribute over others
    return chunkList

def avgChunkAccuracy(chunkList, nodeNumberList,iterations, accuracyFn = accuracy):
    """calculates avg accuracy of nn of given structure using each chunk as a test set"""
    assert(len(chunkList[0][0][0]) == nodeNumberList[0]) #make sure correct number of input nodes
    assert(len(chunkList[0][0][1]) == nodeNumberList[-1]) #make sure correct number of output nodes
    accuracies = []
    for i in range(len(chunkList)):
        testSet = chunkList[i]
        trainingSet = []
        for chunk in chunkList:
            if chunk != chunkList[i]:
                trainingSet += chunk
        nn = NeuralNetwork(nodeNumberList)
        nn.backPropegateLearning(trainingSet,iterations)
        binaryMatrix(nn,testSet)
        accuracies.append(accuracyFn(nn,testSet))
    return mean(accuracies) #return average accuracy across all test sets

def crossValidation2(examples):
    """crossvalidates on examples for varying structures-
    default is single hidden layer with nodes 1-10, but can be changed
    to any desired tests"""
    chunkList = randomChunks(examples,5)
    inputNodes = len(examples[0][0])
    outputNodes = len(examples[0][1])
    hiddenLayerNodes = 1 #beginning # nodes in hidden layer
    #for only 1 hidden layer
    bestStructure = dict()
    for i in range(10):
        nodeNumberList = [inputNodes, hiddenLayerNodes, outputNodes]
        currentAccuracy = avgChunkAccuracy(chunkList, nodeNumberList, 5000)
        hiddenLayerNodes += 1
        print(nodeNumberList)
        print("accuracy:{}".format(currentAccuracy))

def iterationTest(nodeNumberList, pairs):
    """for given network structure and data, performs crossValidation
    using varying iteration number and outputs the dictionary iterations:accuracy"""
    iters = [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500]
    chunkList = randomChunks(pairs)
    accs = dict()
    for i in iters:
        acc = avgChunkAccuracy(chunkList,nodeNumberList,i)
        accs[i] = acc
        print("iterations:{}".format(i))
        print("accuracy:{}".format(acc))
        print("-------------------------------")
    print(accs)


################################################################################
### Main
################################################################################

def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = pairs #[([1.0] + x, y) for (x, y) in pairs]

    #iterationTest([30,3,1],pairs)
    # Check out the data:
    #for example in training:
    #    print(example)
    crossValidation2(training)
    #print(nn)
    """
    #quit()
    #nn = NeuralNetwork([len(training[0][0]),5, len(training[0][1])])
    #nn.backPropegateLearning(training)
    for example in training:
        print("--------------------------------------------------------")
        print("input: {}  ||  correct output: {}".format(example[0],example[1]))
        print()
        fp = nn.forwardPropegate(example[0])
        print("nn output: {}".format(fp))
        print()
        rounded = []
        for i in range(len(fp)):
            rounded.append(round(fp[i]))
        print("predicted class: {}".format(rounded))
        print("--")
        print("correctly classified: "+str(rounded == example[1]))
        print()
        print()
        """


    ##a = NodeNetwork([2,4,6])
    #print(a.inputNodes[0].outPaths)
    #print(a.hiddenLayers)
    #print('hidden layers')
    #for layer in a.hiddenLayers:
    #    print(layer)
    #print(a.hiddenLayers[0][0].inPaths)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    # nn = NeuralNetwork([3, 6, 3])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()

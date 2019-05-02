"""
------------------------------------------------
Names:      Chad Morse and Chris Browne
Date:       5/1/19
Assignment: Project 4
Objective:  Implement forward propagating neural network that learns
            through back propagation
------------------------------------------------
"""

import csv, sys, random, math

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

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
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

    def updateInJandAI(self):
        """dot of weights and inputs-- for this to wrk the previous layer's ai's must be set"""
        inj = 0
        for connection in self.inPaths:
            inj += (connection.i.ai * connection.weight)
        inj += self.bias
        self.inj = inj
        self.ai = logistic(inj)

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
    def __init__(self, nodeNumberList):
        assert(len(nodeNumberList) > 1) #must have at least input layer and output
        hiddenLayers = []
        inputNodes = [InputNode(0, i) for i in range(nodeNumberList[0])]
        outputNodes = [OutputNode(len(nodeNumberList) - 1, i) for i in range(nodeNumberList[-1])]
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

    def gPrime(self, x):        # WE HAVE TO CHANGE THIS, SEE PIAZZA
        return logistic(x)*(1-logistic(x))

    def forwardPropegate(self, input):
        #load input into input layer
        assert(len(input) == len(self.inputNodes)) #error check
        for i in range(len(self.inputNodes)):
            self.inputNodes[i].setInput(input[i])

        #continue down layers
        for layer in self.hiddenLayers + [self.outputNodes]:
            for node in layer:
                node.updateInJandAI()

        #now collect ai of each output node into a list
        output = []
        for node in self.outputNodes:
            output.append(node.ai)
        return output

    def backPropegateLearning(self, inputs):
        # Weights are initialized when connection object is made
        iterations = 5000
        for _ in range(iterations):
            for inp in inputs:
                for i in range(len(self.inputNodes)):
                    self.inputNodes[i].setInput(inp[0][i])
                    # ai ←xi

                for layer in self.hiddenLayers + [self.outputNodes]:
                    for node in layer:
                        node.updateInJandAI()
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

    def __repr__(self):
        return str([self.inputNodes] + self.hiddenLayers + [self.outputNodes])

################################################################################
### Activation Functions
################################################################################

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
        chunks.append(l[i:i+k])
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
            return learner(bestSize, examples)
        printEpoch(size, errorTraining, errorValidation)
        size += 1

def crossValidation(size, k, exampleChunks):
    avgErrorTraining = 0
    avgErrorValidation = 0

    for fold in range(1, k):
        trainingSet, validationSet = partition(exampleChunks, fold)
        hypothesis = learner(size, trainingSet)
        avgErrorTraining += errorRate(hypothesis, trainingSet)
        avgErrorValidation += errorRate(hypothesis, validationSet)
    return avgErrorTraining / k, avgErrorValidation / k


################################################################################
### Main
################################################################################

def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = pairs #[([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    #for example in training:
    #    print(example)
    n = crossValidationWrapper(2,training)
    print(n)
    quit()
    nn = NeuralNetwork([len(training[0][0]),5, len(training[0][1])])
    nn.backPropegateLearning(training)
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

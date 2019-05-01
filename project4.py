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

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

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
        self.bias = 0
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

    def fixWeights(self, learningRate):
        for connection in self.outPaths:
            connection.weight = connection.weight + learningRate * self.ai * connection.j.error

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


class NodeNetwork:
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

    def gPrime(self, x):
        return logistic(x)*(1-logistic(x))

    def backPropegateLearning(self, inputs):
        # Weights are initialized when connection object is made
        iterations = 100
        for _ in range(iterations):
            for inp in inputs:
                for i in range(len(self.inputNodes)):
                    self.inputNodes[i].setInput(inp[0])
                    # ai ←xi

                for layer in self.hiddenLayers + [self.outputNodes]:
                    for node in layer:
                        node.updateInJandAI()
                        # inj← sum(wi,j*ai)
                        # aj ←g(inj)

                for j in range(len(self.outputNodes)):
                    self.outputNodes[j].error = self.gPrime(self.outputNodes[j].inj) * (inp[1] - self.outputNodes[j].aj)
                    # Δ[j]←g′(inj) × (yj − aj)

                for reverseLayer in self.hiddenLayers.reverse() + [self.inputNodes.reverse()]:
                    for revNode in reverseLayer:
                        weights = revNode.getOutWeights()
                        errors = revNode.getOutErrors()
                        dotProd = dot_product(weights, errors)
                        revNode.error = self.gPrime(revNode.inj) * dotProd
                        # Δ[i] ← g′(ini) sum(wi,j Δ[j])

                for connectedLayer in [self.inputNodes] + self.hiddenLayers:
                    for node in connectedLayer:
                        node.fixWeights(self.learningRate)
                        # wi,j←wi,j + α × ai × Δ[j]

            # Change if we go until accuracy level is met
            self.learningRate = self.learningRate - (self.learningRate * (1 / iterations))

        return self


        #NOW IN LOOP for until satisfied
        #   NOw loop through examples
        #       Put input values inputNodes
        #       For layer in (self.hiddenLayers + [outputLayer]):
        #           [set node.values to correct list for all nodes in layer
        #           then call node.updateValue] - updateLayerActivations
        #######Back Propegation
        #       For each node in self.outputNodes:
        #           sets errors of output nodes (use g'(x) = logistic(x)*(1-logistic(x))) inj = value, aj = activation
        #       For layer in reverse(self.hiddenlayers ) + [input] ** make sure this is in right order
        #           set error of each node in layer using


        ###hidden layer node should track both inward and outward weights


def main():
    """
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    for example in training:
        print(example)
    """
    a = NodeNetwork([2,4,6])
    print(a.inputNodes[0].outPaths)
    print(a.hiddenLayers)
    print('hidden layers')
    for layer in a.hiddenLayers:
        print(layer)
    #print(a.hiddenLayers[0][0].inPaths)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    # nn = NeuralNetwork([3, 6, 3])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()

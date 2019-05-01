"""
PLEASE DOCUMENT HERE

Usage: python3 project3.py DATASET.csv
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
### Neural Network code goes here
class Node:
    def __init__(self, layerIndex, inPaths=[], outPaths=[]):
        self.value = 0
        self.activation = 0
        self.error = 0
        self.layerIndex = layerIndex
        self.inPaths = inPaths
        self.outPaths = outPaths

    def __str__(self):
        return "NODE"+str(self.layerIndex)

    def __repr__(self):
        return "NODE"+str(self.layerIndex)

    def initializeWeightsValues(self):
        self.weights = [random.random() for x in range(len(self.inPaths) + 1)]
        self.values = [1 for x in range(len(self.inPaths) + 1)]
        self.errors = [0 for x in range(len(self.outPaths) + 1)]

    def updateValue(self):
        self.value = dot_product(self.weights, self.values)
        self.activation = logistic(self.value)


class InputNode(Node):
    def __init__(self, label, outPaths=[]):
        super().__init__(label,[], outPaths)

class OutputNode(Node):
    def __init__(self, label, inPaths=[]):
        super().__init__(label, inPaths, [])

class NodeNetwork:
    def __init__(self, nodeNumberList):
        assert(len(nodeNumberList) > 1) #must have at least input layer and output
        hiddenLayers = []
        inputNodes = [InputNode(i) for i in range(nodeNumberList[0])]
        outputNodes = [OutputNode(i) for i in range(nodeNumberList[-1])]
        for i in range(len(nodeNumberList) - 2):
            hiddenLayers.append([Node(i) for i in range(nodeNumberList[i+1])])
        #connect input nodes and first hidden layer
        for inputNode in inputNodes:
            inputNode.outPaths = hiddenLayers[0]
        for node in hiddenLayers[0]:
            node.inPaths = inputNodes
        #connect hidden layers
        for i in range(len(hiddenLayers) - 1):
            for node in hiddenLayers[i]:
                node.outPaths = hiddenLayers[i+1]
            for node in hiddenLayers[i+1]:
                node.inPaths = hiddenLayers[i]
        #connect last hidden layer to output nodes
        for node in hiddenLayers[-1]:
            node.outPaths = outputNodes
        for outputNode in outputNodes:
            outputNode.inPaths = hiddenLayers[-1]
        self.inputNodes = inputNodes
        self.hiddenLayers = hiddenLayers
        self.outputNodes = outputNodes

    def initializeNetworkWeights(self):
        for node in self.inputNodes:
            node.initializeWeightsValues()
        for layer in self.hiddenLayers:
            for node in layer:
                node.initializeWeightsValues()
        for node in self.outputNodes:
            node.initializeWeightsValues()

    def backPropegateLearning(self, input):
        #initializeNetwork weights
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
    a = NodeNetwork([3,4,6,5,2])
    print(a.inputNodes[0].outPaths)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    # nn = NeuralNetwork([3, 6, 3])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()

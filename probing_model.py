from torch import nn

class MultiLayerProbingModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, activation=None):
        super(MultiLayerProbingModel, self).__init__()

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(embedding_dim, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        # Output layer, 10 units - one for each digit
        self.fc = nn.Linear(256, num_classes)

        # Define sigmoid activation and softmax output
        #self.relu = nn.ReLU()
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        #x = self.hidden(x)
        #x = self.relu(x)
        x = self.fc(x)
        if  self.activation != None:
            x = self.activation(x)
        return x

class LinearProbingModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, activation=None):
        super(LinearProbingModel, self).__init__()

        self.input = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(p=0.2)

        # Define sigmoid activation and softmax output
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.dropout(x)
        x = self.input(x)
        x = self.activation(x)
        return x

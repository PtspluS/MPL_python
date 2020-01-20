from math import exp, tanh
import random

class Neuronne:

    def __init__(self, input=1, activate='sigmoid'):
        try :
            self.weights = []
            # le b est le poids du bias
            self.b = random.uniform(-0.5, 0.5)
            # on crée un tableau avec
            for i in range(input):
                self.weights.append(random.uniform(-0.5, 0.5))

            self.activation = activate

        except Exception as err:
            print(err)

    def sigmoid(self, x):
        try:
            return 1/(1-exp(x))
        except:
            return 0

    def sigmoid_der(self, x):
        return x * (1-x)

    def heavyside(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def heavyside_der(self,x):
        return 0

    def hyperbolique(self, x):
        return tanh(x)

    def hyperbolique_der(self, x):
        return 1 - tanh(x)*tanh(x)

    def softmax(self, x):
        if x > 0.5:
            return 1
        else:
            return 0

    def relu(self, x):
        return max(0, x)

    def relu_der(self, x):
        return 1 * (x > 0)

    def calculate(self, input_data):
        try:
            out = 0
            for i in range(len(self.weights)):
                out += input_data[i] * self.weights[i]
            out += self.b
            if self.activation == 'heavyside':
                return self.heavyside(out)
            elif self.activation == 'sigmoid':
                return self.sigmoid(out)
            elif self.activation == 'tanh':
                return self.hyperbolique(out)
            elif self.activation == 'softmax':
                return self.softmax(out)
            elif self.activation == 'relu':
                return self.relu(out)
            else:
                raise NameError('Invalid activation name')
        except Exception as e:
            print(e)

    def derivative(self, x):
        try:
            if self.activation == 'sigmoid':
                return self.sigmoid_der(x)
            elif self.activation =='tanh':
                return self.hyperbolique_der(x)
            elif self.activation == 'heavyside':
                return self.heavyside_der(x)
            elif self.activation == 'softmax':
                pass
            elif self.activation =='relu':
                return self.relu_der(x)
            else:
                raise NameError('Invalid activation name')
        except Exception as err:
            print(err)


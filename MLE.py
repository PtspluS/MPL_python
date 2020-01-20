from Neuronne import Neuronne
import pickle

class Model:

    def __init__(self, input_size=1):
        self.neuronnes = []
        self.couches_layers = []
        self.input_size = input_size

        self.couches = 0

    def add(self, nb_neuronnes=1, activate='sigmoid'):
        layer = []
        for i in range(nb_neuronnes):
            if self.couches == 0:
                neuronne = Neuronne(input=self.input_size, activate=activate)
            else:
                neuronne = Neuronne(len(self.couches_layers[self.couches-1]), activate=activate)
            layer.append(neuronne)
            self.neuronnes.append(neuronne)

        self.couches_layers.append(layer)
        self.couches += 1

    def predict(self, data):
        inputs = []
        for l in range(len(self.couches_layers)):
            if l ==0:
                inputs = data
            outputs = []
            for n in (self.couches_layers[l]):
                outputs.append(n.calculate(inputs))
            inputs = outputs

        return outputs

    def delta_output(self, output, expected):
        return output*(1-output)*(expected-output)

    def delta_hidden(self, output, excpected):
        pass

    # TODO find where is the error
    def fit(self, data, target, learning_rate=0.3, epochs=1, show=1):
        #try:
            nb_egals = 24
            for current_epoch in range(epochs):
                for data_index in range(len(data)):
                    inputs = []
                    # tableau avec toutes les sorties
                    outputs_tab = []
                    # on calcul la sortie o
                    # on passe dans tous les layers
                    for l in range(len(self.couches_layers)):
                        # on scrute le premier layer
                        if l == 0:
                            inputs = data[data_index]
                        outputs = []
                        # pour chaque neuronnes de la couche
                        for n in (self.couches_layers[l]):
                            outputs.append(n.calculate(inputs))
                        outputs_tab.append(outputs)
                        inputs = outputs
                    # o = outputs[0]
                    delta_tab = []
                    i = 0
                    # for each layer from end to start
                    # back propagation
                    for la in range(len(self.couches_layers), 0, -1):
                        deltas = []
                        if la == len(self.couches_layers):
                            for o in outputs_tab[la-1]:
                                der = self.couches_layers[la-1][0].derivative(o)
                                x = der*(target[data_index]-o)
                                deltas.append(x)
                            delta_tab.append(deltas)
                            i += 1
                        else:
                            for n in range(len(self.couches_layers[la-1])):
                                o = outputs_tab[la-1][n]
                                dw = []
                                d = 0
                                for w in range(len(self.couches_layers[la])):
                                    dw.append(delta_tab[len(outputs_tab)-1-la][w]*self.couches_layers[la][w].weights[n])
                                dw.append(delta_tab[len(outputs_tab)-1-la][w]*self.couches_layers[la][w].b)
                                der = self.couches_layers[la][0].derivative(o)
                                delta = der*sum(dw)
                                deltas.append(delta)
                            delta_tab.append(deltas)
                            i += 1
                    # print(delta_tab)
                    # forward propagation
                    delta_tab = [delta_tab[x] for x in range(len(delta_tab)-1, -1, -1)]
                    for l_index in range(len(self.couches_layers)):
                        # pour chaque neuronnes de la couche
                        for n_index in range(len(self.couches_layers[l_index])):
                            for w in self.couches_layers[l_index][n_index].weights:
                                if l_index == 0:
                                    w -= learning_rate * delta_tab[l_index][n_index] * data[data_index][n_index]
                                else :
                                    w -= learning_rate*delta_tab[l_index][n_index]*outputs_tab[l_index][n_index]
                            self.couches_layers[l_index][n_index].b = self.couches_layers[l_index][n_index].b - learning_rate*delta_tab[l_index][n_index]*outputs_tab[l_index][n_index]

                print("Epoch : "+str(current_epoch)+'/'+str(epochs))
                if show == 1:
                    print('['+'='*round(((current_epoch/epochs)*nb_egals))+'>'+'.'*round(((1-current_epoch/epochs)*nb_egals))+']'+' Training')
            print("Epoch : " + str(current_epoch+1) + '/' + str(epochs))
            if show == 1:
                print('['+'='*nb_egals+'>'+']'+' Training Finished')

            # calcul de la precision du model
            precision = 0
            for d, t in zip(data, target):
                if self.predict(d)[0] == t:
                    precision += 1.0
            precision = precision/len(target)

            # retour des data de l'entrainement
            return precision

        #except Exception as err:
        #    print(err)

    def __str__(self):
        connection = 0
        nb_egal = 30
        string = ""
        string += "Nombre de neuronnes au total : " + str(len(self.neuronnes))+'\n'
        string += "=" * nb_egal + '\n'
        for l in self.couches_layers:
            string += "Nombre de neuronnes sur la couche : " + str(len(l)) + '\n'
            string += "Activation fonction : "+str(l[0].activation) + '\n'
            string += "=" * nb_egal+ '\n'

        for i in range(len(self.couches_layers) - 1):
            connection += len(self.couches_layers[i]) * (len(self.couches_layers[i + 1]))

        string += "Nombre de connections : " + str(connection) + '\n'
        return string

    def save(self, name='ai'):
        try:
            with open(name, 'rw') as file:
                pickle.dump(self, file)
        except Exception as err:
            print(err)
        finally:
            file.close()

    def load(self, name='ai'):
        try:
            with open(name, 'rw') as file:
                obj = pickle.load(file)
                return obj
        except Exception as err:
            print(err)

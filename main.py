from MLE import Model

""""
Tuto :
Pour créer un model :
    model = Model
    
Pour ajouter une couche de neuronnes :
    model.add(<nb_neuronnes>, <fonction_d'activation>)
    
    <nb_neuronnes> :
        int : Nombres de neuronnes sur la couche, la première couche doit avoir autant de neuronnes que le nombre d'inputs datas
    
    <fonction_dactivation> :
        string : Nom de la fonction d'activation, choix entre :
            - heavyside
            - sigmoid
            - tanh
            - softmax
            - relu
            
Pour entrainer le reseau :
    model.fit(<data>, <target_data>, <learning_rate>, <epochs>)
    return : 
        float : pourcentage de precission a la fin de l'entrainement
    
    <data> : 
        list[list] : tableau de tableau de data pour entrainer
        
    <target> : 
        list : tableau de resultats attendu
        
    <learning_rate> :
        float : vitesse d'apprentissage
        
    <epochs> : 
        int : nombre de passage de l'algo pour apprendre
        
        
Pour predire un resultat :
    model.predict(<data>)
    return :
        tab[float] : resultat de la prediction
    
    <data> : 
        list : donnes a predire
"""

model = Model(input_size=2)
model.add(5)
model.add(3)
model.add(2)
model.add(1, 'relu')

training_data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
target_data = [
    0,
    1,
    1,
    0
]
print(model)

print(model.predict(training_data[0]))
pr = model.fit(training_data, target_data, 0.001, 1000)
print(model.predict(training_data[0]))
print(model.predict(training_data[1]))
print(model.predict(training_data[2]))
print(model.predict(training_data[3]))

print('Model train with '+str(pr*100)+'% of accuracy')


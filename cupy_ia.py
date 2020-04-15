#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cupy as cp
import numpy as np
from time import time

def sigmoid(x):
    """La fonction sigmoïde est une courbe en S:
    https://fr.wikipedia.org/wiki/Sigmo%C3%AFde_(math%C3%A9matiques)
    """

    return 1 / (1 + cp.exp(-x))

def sigmoid_prime(z):
    """La dérivée de la fonction sigmoid,
    soit sigmoid' comme f' !
    """

    return z * (1 - z)

def relu(x):
    """Rectifie les négatifs à 0:
    -1 > 0
     1 > 1
     Rectified Linear Unit:

    In the context of artificial neural networks, the rectifier is an
    activation function defined as the positive part of its argument.
    https://bit.ly/2HyT4ZO sur Wikipedia en.
     """

    return cp.maximum(0, x)

def relu_prime(z):
    """Fonction: 1 pour tous les réels positifs ou nuls et 0 pour les réels négatifs.

    La fonction de Heaviside (également fonction échelon unité, fonction
    marche d'escalier) est la fonction indicatrice de R.
    Une fonction fonction indicatrice, est une fonction définie sur un
    ensemble E qui explicite l’appartenance ou non à un sous-ensemble F de E
    de tout élément de E.
    """

    return cp.asarray(z > 0, dtype=cp.float32)


class SemaphoreIA:
    """Réseau de neuronnes Perceptron multicouches."""

    def __init__(self, train, learningrate):
        """train = Nombre de shot pour l'apprentissage
        learningrate = coeff important
        """

        print("Calcul avec cupy ... cupy ... cupy ... cupy ... ")
        
        self.train = train
        self.learningrate = learningrate

        # Réseau de neurones: colonne 1600 en entrée, 2 nodes de 100, sortie de 27 caractères
        self.layers = [1600, 100, 100, 27]
        # Fonction d'activation: imite l'activation d'un neuronne
        self.activations = [relu, relu, sigmoid]
        self.weight_list = None

        fichier = cp.load('./semaphore.npz')

        a = cp.array(fichier['x_train'])
        b = cp.array(fichier['y_train'])
        c = cp.array(fichier['x_test'])
        d = cp.array(fichier['y_test'])
        print(type(a))

        self.x_train = a
        self.y_train = b
        self.x_test = c
        self.y_test = d
        
        print(type(self.x_train))
        a = "Training: Shot {} Lettre {}; Testing: Shot {} Lettre {}"
        print(a.format( len(self.x_train), len(self.y_train),
                        len(self.x_test),  len(self.y_test)))

    def training(self):
        """Apprentissage avec 60 000 images
        Poids enregistré dans weights_cupy.npy
        """

        print("Training...")

        # Matrice diagonale de 1
        diagonale = cp.eye(27, 27)

        # globals() Return a dictionary representing the current global symbol table.
        self.activations_prime = [globals()[fonction.__name__ + '_prime'] for fonction in self.activations]

        node_dict = {}

        # Liste des poids
        # Initialisation des poids des nodes, pour ne pas à être à 0
        # Construit 3 matrices (100x1600, 100x100, 27x100)
        # /cp.sqrt() résultat expérimental de l'initialisation de Xavier Glorot et He
        weight_list = [cp.random.randn(self.layers[k+1], self.layers[k]) / \
                       cp.sqrt(self.layers[k]) for k in range(len(self.layers)-1)]

        # vecteur_ligne = image en ligne à la 1ère itération
        # nombre_lettre = nombre correspondant à la lettre de l'image
        # i pour itération, vecteur_colonne = x_train de i, nombre_lettre = y_train de i
        for i, (vecteur_ligne, nombre_lettre) in enumerate(zip(self.x_train, self.y_train)):

            # la ligne devient colonne
            vecteur_colonne = cp.array(vecteur_ligne, ndmin=2).T

            # IndexError: arrays used as indices must be of integer or boolean type.
            # (actual: <class 'numpy.object_'>) in diagonale[:,[nombre_lettre]]
            nombre_lettre = int(nombre_lettre)
            
            # Forward propagation
            node_dict[0] = vecteur_colonne
            for k in range(len(self.layers)-1):
                # weight_list[k] (100x1600, 100x100 27x100) vecteur_colonne (1600,)
                # z de format 100 x 1
                z = cp.dot(weight_list[k], vecteur_colonne)

                # self.activations = non linéaire sinon sortie fonction linéaire de l'entrée
                # imite le seuil d'activation électrique du neuronne
                vecteur_colonne = self.activations[k](z)

                node_dict[k+1] = vecteur_colonne

            # Retro propagation, delta_a = écart entre la sortie réelle et attendue
            delta_a = vecteur_colonne - diagonale[:,[nombre_lettre]]
            
            # Parcours des nodes en sens inverse pour corriger proportionnellement
            # les poids en fonction de l'erreur par rapport à la valeur souhaitée
            # Descente du Gradient stochastique
            for k in range(len(self.layers)-2, -1, -1):
                delta_z = delta_a * self.activations_prime[k](node_dict[k+1])
                delta_w = cp.dot(delta_z, node_dict[k].T)
                delta_a = cp.dot(weight_list[k].T, delta_z)
                # Pour converger vers le minimum d'erreur
                weight_list[k] -= self.learningrate * delta_w

        self.weight_list = weight_list
    
        # Dans un fichier
        print("type(weight_list :)", type(weight_list),
                "\nlen(weight_list) =", len(weight_list),
                "\n    0", len(weight_list[0]), type(weight_list[0]),
                "\n    1", len(weight_list[1]), type(weight_list[1]),
                "\n    2", len(weight_list[2]), type(weight_list[2]))

        cp.save('./weights_cupy.npy', weight_list, allow_pickle=True)
        print('weights_cupy.npy enregistré')
        
    def testing(self):
        """Teste avec les images de testing, retourne le ratio de bon résultats"""

        print("Testing...")

        # Avec sauvegarde
        weight_list = np.load('weights_cupy.npy', allow_pickle=True)
        
        # Nombre de bonnes reconnaissance
        success = 0

        for vecteur_ligne, nombre_lettre in zip(self.x_test, self.y_test):
            
            vecteur_ligne = cp.array(vecteur_ligne)
            nombre_lettre = cp.array(nombre_lettre) 
        
            for k in range(len(self.layers)-1):
                vecteur_ligne = self.activations[k](cp.dot(weight_list[k], vecteur_ligne))

            reconnu = cp.argmax(vecteur_ligne)
            if reconnu == nombre_lettre:
                success += 1

        if len(self.x_test) != 0:
            resp = 100.0 * success / len(self.x_test)
        else:
            resp = 0
        return resp


def main():
    train = 60000
    learningrate = 0.023
    sia = SemaphoreIA(train, learningrate)
    sia.training()
    resp = sia.testing()
    print("Learningrate: {} Résultat {}".format(learningrate, round(resp, 1)))



if __name__ == "__main__":

    t = time()
    main()
    print("Calcul en:", round((time() - t), 1), "s")

"""
cupy 7.3
Successfully installed cupy-7.3.0 fastrlock-0.4
Nvidia 1060 GTX

Calcul avec cupy ... cupy ... cupy ... cupy ... 
Training: Shot 60000 Lettre 60000; Testing: Shot 10000 Lettre 10000
Training...
type(weight_list :) <class 'list'> 
len(weight_list) = 3 
    0 100 <class 'cupy.core.core.ndarray'> 
    1 100 <class 'cupy.core.core.ndarray'> 
    2 27 <class 'cupy.core.core.ndarray'>
weights_cupy.npy enregistré
Testing...
Learningrate: 0.023 Résultat 89.9
Calcul en: 91.7 s
"""

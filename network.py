"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object): 

    def __init__(self, sizes): 
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) # toma el número de capas que se considerarán
        self.sizes = sizes # se guardan los tamaños de cada capa
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #se asigna aleatoriamente los biases para cada conexión
        self.weights = [np.random.randn(y, x) #se asignan aleatoriamente los pesos para cada neurona
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):  # se juntan el vector de biases y de pesos y se toman un dato de cada vector en cada iteración
            a = sigmoid(np.dot(w, a)+b) #calculamos la activacion de la RNA
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:  # si se da un conjunto de datos se ejecutan las siguientes 2 lineas 
            test_data = list(test_data) #se convierte el conjunto de datos en lista
            n_test = len(test_data) # se toma el tamaño de la lista

        training_data = list(training_data) # se convierten los datos de entrenamiento en lista
        n = len(training_data) # se toma el tamaño de la lista

        for j in range(epochs): #se recorre el número de epocas dada
            random.shuffle(training_data) # reorganizamos los datos de entrenamiento
            mini_batches = [
                training_data[k:k+mini_batch_size] # se construye el vector de mini-batches apartir de una particion del conjunto de datos de entrenamiento
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: # Se recorren los mini-batches
                self.update_mini_batch(mini_batch, eta) # se actualiza el mini-batch
            if test_data: 
                print("Epoch {0}: {1} / {2}".format( # se muestra la época en la que va el conteo y también la exactitud de la predicción
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j)) # si se completa el proceso, se muestra en pantalla junto con la época

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] # se crea la lista de arreglos usando el tamaño de los biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # se crea la lista de arreglos usando el tamaño de los pesos
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # se calculan los gradientes de la funcion de costo
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # se actualizan los valores de nabla b
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # se actualizan los valores de nabla w
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] # se calculan nuevamente los pesos 
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] # se calculan nuevamente los biases

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] # se crea la lista de arreglos usando el tamaño de los biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # se crea la lista de arreglos usando el tamaño de los pesos
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights): # se juntan el vector de biases y de pesos y se toman un dato de cada vector en cada iteración
            z = np.dot(w, activation)+b # se calcula el valor de z con los datos de b y w
            zs.append(z) # se añade el valor de z al vector de z's
            activation = sigmoid(z) # se calcula la activacion
            activations.append(activation) # se añade la activacion al vector de activaciones
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) # se calcula delta, multiplicando la derivada de la funcion de costo con la derivada de la funcion sigmoide
        nabla_b[-1] = delta # se guarda el valor de delta en biases
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # se calcula el peso multiplicando dela con el vector de activaciones
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): # se recorre el numero de capas a partir de la segunda
            z = zs[-l] # tomamos el ultimo valor de z
            sp = sigmoid_prime(z) # se calcula la derivada de la funcion sigmmoide en z
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # se calcula el valor de delta con los pesos y el delta anterior
            nabla_b[-l] = delta # se guarda el valor de delta en biases
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # se calcula el ultimo peso con delta y las activaciones
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) #Se toma un vector de valores de test_data en el cual a la primera entrada se calcula su activacion
                        for (x, y) in test_data]            # y se calcula la activación máxima
        return sum(int(x == y) for (x, y) in test_results)  # se suma la cantidad de veces en que se tiene resultados correctos 

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) # al vector de activaciones se le resta el vector de salidas y eso es la derivada parcial de la funcion de costo
                                      # con respecto de a 

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z)) #Calculo de la funcion sigmoide para z

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) #Se calcula la derivada de la función sigmoide

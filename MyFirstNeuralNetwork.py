from typing import Any

import numpy as np
import logging as logging

from numpy import dtype, float64, ndarray

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

log = logging.getLogger(__name__)

def create_random_number_generator():
    # e un generatore di numeri casuali
    # con il seed (valore iniziale) impostato a 42
    rng = np.random.default_rng(42)
    log.info("Random number generator initialized with seed 42, number: %s", rng)
    return rng

def create_input():
    # Creazione della matrice degli input
    # X = [
    #       [0, 1],
    #       [1, 0]
    #               ]
    X = np.array([[0., 1.], [1., 0.]], dtype=np.float64)
    log.info("Input data initialized: %s", X)
    return X

def create_weights(rng):
    # 0 è la media
    avg = 0
    # 0.5 è la deviazione standard
    std = 0.5
    # size=(2, 1) indica che vogliamo una matrice di pesi con 2 righe e 1 colonna
    W = rng.normal(avg, std, size=(2, 1))
    # W = [w1, w2]
    log.info("Weights initialized: %s", W)
    return W

def create_bias():
    # Crea un array con un solo elemento
    # il primo elemento è 0
    b = np.zeros(1)
    log.info("Bias initialized: %s", b)
    return b

def init_neural_network():
    # rng è un random number generator
    rng = create_random_number_generator()
    # Creazione degli input
    X = create_input()
    # Creazione dei pesi
    W = create_weights(rng)
    # Creazione del bias
    b = create_bias()

    return rng, X, W, b

# Passo 1 della regola della catena
# Forward pass
def forward_pass(X, W, b) -> ndarray[tuple[Any, ...], dtype[float64]]:
    # Calcolo output della rete neurale
    # Qui si esegue un prodotto matriciale
    # Le forme sono compatibili
    # (2,2)@(2,1) = (2,1)
    # Permette di ottenere un risultato per ogni evento
    # XW = [
    #       [0w1 + 1w2],
    #       [1w1 + 0w2]
    #       ] = [w2, w1]
    # Poi aggiungi il bias a entrambe le righe
    # z = [ [w2 + b],
    #       [w1 + b]
    # ]
    z = X @ W + b
    log.info("Forward pass result: %s", z)
    # Questa è la sigmoide
    # sigma(z)=1/(1+e^(-z))
    y_hat = 1.0 / (1.0 + np.exp(-z))
    log.info("Forward pass output: %s", y_hat)

    return y_hat

"""
Costruzione di un singolo neurone con 2 ingressi:
    - Il calcolo che viene eseguito e':
        - z = x1*w1 + x2*w2 + b
        - y(capuccio) = sigma(z) = 1/(1 + e^(-z)) 
"""
if __name__ == '__main__':
    rng, X, W, b = init_neural_network()

    forward_pass(X, W, b)
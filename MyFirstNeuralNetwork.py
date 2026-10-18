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


# Per continuare con il laboratorio
# Bisogna calcolare i gradienti tramite la backpropagation
#   - Prima di iniziare l'addestramento dobbiamo introdurre una loss
#   - La loss è il numero che misura quanto le previsioni sono lontane dalle risposte corrette
#   - La Loss usata per l'implementazione manuale è la Mean Squared Error o MSE
#   - L = 1/N * Σ(sommatoria con i=1 fino a N)(y_hat(i) - y(i))²
def mse_loss(y_hat, Y):
    # Differenza tra ogni previsione e la risposta corretta
    errors = y_hat - Y
    log.info("Errors calculated: %s", errors)

    # Quadrato di ogni errore
    squared_errors = errors ** 2
    log.info("Squared errors calculated: %s", squared_errors)

    # Media: un solo numero che misura l'errore complessivo
    loss = np.mean(squared_errors)
    log.info("Mean Squared Error (MSE) loss calculated: %s", loss)
    return loss

def backward_pass(X, Y, y_hat):
    # Dalla loss alle previsioni
    # Per la MSE, la derivata rispetto ogni previsione è
    # dL/dy_hat = 2/N * (y_hat - Y)
    dL_dy_hat = ((2.0 / Y.size) * (y_hat - Y))

    # Attraverso la sigmoide
    # La derivata della sigmoide si può calcolare direttamente dalla sua uscita
    # sigma'(z) = sigma(z) * (1 - sigma(z))
    dy_hat_dz = (y_hat * (1.0 - y_hat))

    # Applichiamo la regola della catena
    dL_dz = dL_dy_hat * dy_hat_dz

    # Da z ai pesi e al bias
    # Dato che z = XW + b, i gradienti sono
    dL_dW = X.T @ dL_dz
    dL_db = np.sum(dL_dz, axis=0)

    log.info("Gradiente dei pesi:\n%s", dL_dW)
    log.info("Gradiente del bias: %s", dL_db)

    return dL_dW, dL_db

def update_parameters(W, b, dW, db, learning_rate):
    W_new = (W - (learning_rate * dW))
    b_new = (b - (learning_rate * db))

    log.info("Updated weights: %s", W_new)
    log.info("Updated bias: %s", b_new)

    return W_new, b_new
"""
Costruzione di un singolo neurone con 2 ingressi:
    - Il calcolo che viene eseguito e':
        - z = x1*w1 + x2*w2 + b
        - y(capuccio) = sigma(z) = 1/(1 + e^(-z)) 
"""
if __name__ == '__main__':
    rng, X, W, b = init_neural_network()
    Y = np.array([[1.], [1.]], dtype=np.float64)
    learning_rate = 0.1
    epochs = 100
    loss = 0

    for epoch in range(1, epochs + 1):
        # Previsione con i parametri attuali
        y_hat = forward_pass(X, W, b)
        # Errore delle previsioni
        loss = mse_loss(y_hat, Y)
        # Gradienti calcolati con i parametri attuali
        dW, db = backward_pass(X, Y, y_hat)
        # Aggiornamento dei parametri
        W, b = update_parameters(W, b, dW, db, learning_rate)

        if epoch == 1 or epoch % 10 == 0:
            log.info("Epoch %d | Loss before step: %6f", epoch, loss)
        log.info("Epoch %d | Weights: %s | Bias: %s", epoch, W, b)
        log.info("-----------------------------------------------------------")
    # Verifico l'effetto dell'aggiornamento
    y_hat_final = forward_pass(X, W, b)
    loss_final = mse_loss(y_hat_final, Y)

    log.info(
        "Loss prima dell'ultimo aggiornamento: %.8f | Loss dopo: %.8f",
        loss,
        loss_final,
    )
from idlelib.iomenu import errors

import numpy as np
import logging as logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

log = logging.getLogger(__name__)

def create_random_number_generator():
    rng = np.random.default_rng(42)
    return rng

def create_Y():
    Y = np.array([[0.], [1.], [1.], [0.]], dtype=np.float64)
    return Y

def create_input():
    input_x_array = []
    for i in range(0, 2):
        for j in range(0,2):
            input_x_array.append((float(i), float(j)))

    X = np.array(input_x_array, dtype=np.float64)

    return X

def create_W1_and_b1(rng,
                   number_of_entrance,
                   number_of_hidden_states):
    avg = 0
    std = 0.5
    W1 = rng.normal(avg, std, size=(number_of_entrance,number_of_hidden_states))
    b1 = np.zeros(number_of_hidden_states)
    return W1, b1

def create_W2_and_b2(rng,
                   number_of_entrance,
                   number_of_hidden_states):
    avg = 0
    std = 0.5
    W2 = rng.normal(avg, std, size=(number_of_hidden_states,number_of_entrance))
    b2 = np.zeros(number_of_entrance)
    return W2, b2


def init_neural_network():
    rng = create_random_number_generator()
    X = create_input()
    W1, b1 = create_W1_and_b1(rng, 2, 8)
    W2, b2 = create_W2_and_b2(rng, 1, 8)
    return rng, X, W1, b1, W2, b2

def forward_pass(X, W1, b1, W2, b2):
    z_hidden = X @ W1 + b1
    hidden = np.maximum(z_hidden, 0.0)

    z_output = hidden @ W2 + b2
    y_hat = (1.0 / (1.0 + np.exp(-z_output)))

    return y_hat, z_hidden, hidden

def mse_loss(y_hat, Y):
    errors = y_hat - Y
    squared_errors = errors ** 2
    loss = np.mean(squared_errors)
    return loss

def backward_pass(
        X,
        Y,
        y_hat,
        W2,
        hidden,
        z_hidden):
    # Loss -> previsioni
    dL_dy_hat = ((2.0 / Y.size) * (y_hat - Y))
    # Attraversiamo la sigmoid
    dL_dz_output = ((dL_dy_hat * y_hat) * (1.0 - y_hat))


    # Gradienti del secondo strato
    # Parametri dello strato di uscita
    dW2 = hidden.T @ dL_dz_output
    db2 = np.sum(dL_dz_output, axis=0)

    # Risaliamo alle attivazioni nascoste
    dL_dhidden = dL_dz_output @ W2.T
    # Attraversiamo la ReLU
    dL_dz_hidden = dL_dhidden * (z_hidden > 0.0)

    # Parametri dello strato nascosto
    dW1 = X.T @ dL_dz_hidden
    db1 = np.sum(dL_dz_hidden, axis=0)

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1_new = W1 - learning_rate * dW1
    b1_new = b1 - learning_rate * db1
    W2_new = W2 - learning_rate * dW2
    b2_new = b2 - learning_rate * db2
    return W1_new, b1_new, W2_new, b2_new

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    rng, X, W1, b1, W2, b2 = init_neural_network()
    Y = create_Y()
    learning_rate = 0.1
    epochs = 5000

    log.info("Forme: X=%s, Y=%s", X.shape, Y.shape)
    log.info("Strato nascosto: W1=%s, b1=%s", W1.shape, b1.shape)
    log.info("Strato di uscita: W2=%s, b2=%s", W2.shape, b2.shape)

    for epoch in range(1, epochs + 1):
        y_hat, z_hidden, hidden = forward_pass(X, W1, b1, W2, b2)
        loss = mse_loss(y_hat, Y)
        if epoch == 1:
            loss_initial = loss

        dW1, db1, dW2, db2 = backward_pass(
            X=X,
            Y=Y,
            y_hat=y_hat,
            W2=W2,
            hidden=hidden,
            z_hidden=z_hidden,
        )

        W1, b1, W2, b2 = update_parameters(
            W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate
        )

        if epoch == 1 or epoch % 500 == 0:
            log.info("Epoca %d | Loss prima del passo: %.8f", epoch, loss)

    y_hat_final, _, _ = forward_pass(X, W1, b1, W2, b2)
    loss_final = mse_loss(y_hat_final, Y)
    predictions = (y_hat_final >= 0.5).astype(int)

    for x, target, probability, prediction in zip(X, Y, y_hat_final, predictions):
        log.info(
            "Input: %s | Atteso: %d | Output: %.4f | Predetto: %d",
            x, int(target[0]), float(probability[0]), int(prediction[0]),
        )

    accuracy = np.mean(predictions == Y)
    log.info("Loss iniziale: %.8f | Loss finale: %.8f", loss_initial, loss_final)
    log.info("Accuratezza sui quattro esempi di training: %.2f%%", accuracy * 100)


if __name__ == "__main__":
    main()
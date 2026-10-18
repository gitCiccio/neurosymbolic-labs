"""
riconoscere se tre bit non sono tutti uguali, con un MLP in NumPy.
La rete deve restituire:
- 0 se i bit sono tutti uguali;
- 1 se almeno un bit è diverso dagli altri.

| Input | Risposta corretta |
|---|---:|
| `[0, 0, 0]` | 0 |
| `[0, 0, 1]` | 1 |
| `[0, 1, 0]` | 1 |
| `[0, 1, 1]` | 1 |
| `[1, 0, 0]` | 1 |
| `[1, 0, 1]` | 1 |
| `[1, 1, 0]` | 1 |
| `[1, 1, 1]` | 0 |

La consegna
Crea un file NeuralNetworkThreeBits.py e realizza questi passaggi:
1. Genera gli otto input con dei cicli, senza scriverli tutti a mano. (fatto)
2. Genera le etichette controllando se i tre bit sono uguali. (fatto)
3. Costruisci un MLP con 3 ingressi → 8 neuroni nascosti con ReLU → 1 uscita con sigmoide. (fatto)
4. Implementa forward pass, MSE, backpropagation e aggiornamento dei otto parametri. (fatto)
5. Addestra la rete e stampa loss iniziale, loss finale e accuratezza. (fatto)
6. Per ogni input, mostra risposta attesa, output della sigmoide e classe predetta. (fatto)
Puoi partire con seed 42, learning rate 0.1 e 5.000 epoche. Sono valori iniziali da sperimentare, non una garanzia di convergenza.
"""
import logging

import torch
from torch import nn

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_input():
    """
    Qui stiamo realizzando il nostro insieme di input
    Per generalo in automatico utilizziamo 3 for innestati che partono da 0 fino a n(2)
    l'output attesto è il seguente
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 1., 0.],
     [0., 1., 1.],
     [1., 0., 0.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.]]
    :return: qui ritorna un tensore, una matrice del tipo X.shape(8, 3)
    """
    X = []
    max_range = 2
    for i in range(max_range):
        for j in range(max_range):
            for k in range(max_range):
                X.append((float(i), float(j), float(k)))

    return torch.tensor(X, dtype=torch.float32)

def create_Y(X):
    """
    Calcolo dell'output atteso
    [0.,
    1.,
    1.,
    1.,
    1.,
    1.,
    1.,
    .0]
    :param X: Dati in input utilizzati per calcolare l'output Y
    :return: tensore Y, matrice di questo formato Y.shape(8, 1)
    """
    Y = []
    for i in range(len(X)):
        equals = True
        for j in range(len(X[i])-1):
            if X[i][j] != X[i][j+1]:
                equals = False
        if not equals:
            Y.append([1.])
        else:
            Y.append([0.])

    return torch.tensor(Y, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_outputs: int):
        super().__init__()

        self.hidden_layer = nn.Linear(in_features, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        """
        La funzione forward è la funzione che descrive come la rete
        trasforma gli input in una previsione
        :param x: Matrice degli input
        :return: previsione y_hat
        """

        # Lo strato nascosto contiene 8 neuroni gli in_features
        # Dove ogni neurone riceve i tre bit e ne calcola una propria
        # somma pesata
        # z[j] = w[j,1]*x[1] + w[j,2]*x[2] + w[j,3]*x[3] + b[j]
        z_hidden = self.hidden_layer(x)
        # Qui applichiamo la ReLU a ognuno degli 8 valori
        # ReLU(z) = max(0,z)
        # Qui avvengono le attivazioni nascoste
        # Si tratta di una nuova rappresentazione dell'input
        # costruita dalla rete
        hidden = torch.relu(z_hidden)

        # Il neurone finale ottiene le n attivazioni
        # e ne calcola una nuova somma pesata
        # Ora varemo un solo valore che può essere positivo o negativo
        z_output = self.output_layer(hidden)

        # Poi è la sigmoide a portare quel valore compreso tra 0 e 1
        # Questa è la previsione ed è l'output che confronteremo
        # con l'etichetta corretta Y
        y_hat = torch.sigmoid(z_output)

        return y_hat
def main():
    X = create_input()
    Y = create_Y(X)


    in_features = X.shape[1]  # 3 caratteristiche per esempio
    n_hidden = 8  # numero di neuroni scelto da noi
    n_outputs = 1
    rng = torch.manual_seed(42)
    learning_rate = 0.1
    mlp = MLP(
        in_features=in_features,
        n_hidden=n_hidden,
        n_outputs=n_outputs
    )

    # Qui si esegue il calcolo sulla loss
    loss_fn = nn.MSELoss()
    # optimizer è l'oggetto che aggiornerà i parametri
    # mlp.parameters() fornisce pesi e bias di entrambi gli strati
    # lr imposta il learning rate
    # con questa configurazione viene aggiornato secondo
    # Pnuovo = Pvecchio - learning_rate * gradient
    # Questa è la riga che prepara l'ottimizzazione
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
    # qui impostiamo le epoche per l'allenamento
    epochs = 5000
    # qui impostiamo la rete in modalità di allenamento
    mlp.train()

    for epoch in range(1, epochs + 1):
        # zero_grand() azzera i gradienti del passo precedente
        optimizer.zero_grad()

        # Qui stiamo passando l'insieme degli input X
        # al modello assegnato a y_hat
        # eseguendo la chiamata forward su tutti gli esempi
        y_hat = mlp(X)

        # confronto tra le precisioni e le etichette
        # Creando un tensore scalare
        loss = loss_fn(y_hat, Y)
        if epoch == 1:
            # salvataggio della loss iniziale come riferimento
            loss_initial = loss.item()
        # calcolo delle derivate della loss rispetto ai pesi e ai bias
        # utilizzando il grafo ricostruito nella forward
        loss.backward()

        # usa i gradienti calcolati per aggiornare i parametri
        # aggiorna i pesi e i bias della rete neurale
        # non ricalcola la loss esistente
        optimizer.step()
        if epoch == 1 or epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    # Qui impostiamo la rete in modalità di valutazione

    mlp.eval()
    # Stiamo dando i valori alle variabili qui?
    # spiegami l'intero blocco with
    with torch.no_grad():
        # calcolo della nuova previsione con i parametri aggiornati
        y_hat_final = mlp(X)
        # calcolo della nuova revisione usando i parametri aggiornati
        loss_final = loss_fn(y_hat_final, Y).item()

        # calcola la MSE finale ed estrae il numero di pytorch
        predictions = (y_hat_final > 0.5).to(torch.int64)

        # Confronta predizioni ed etichette, converte le risposte corrette in 1.0 e quelle errate in 0.0, poi ne calcola la media. Per esempio, sette risposte corrette su otto producono:
        # \[
        # \text{accuracy}=\frac{7}{8}=0.875
        # \]
        # Moltiplicando per 100 stampi 87.5%.
        accuracy = (predictions == Y).float().mean().item()

    for x, target, probability, prediction in zip(X, Y, y_hat_final, predictions):
        log.info(
            "Input: %s | Atteso: %d | Output: %.4f | Predetto: %d",
            x.tolist(), int(target.item()), probability.item(), prediction.item(),
        )
    log.info("Loss iniziale: %.8f | Loss finale: %.8f", loss_initial, loss_final)
    log.info("Accuratezza sui quattro esempi di training: %.2f%%", accuracy * 100)

if __name__ == '__main__':
    main()
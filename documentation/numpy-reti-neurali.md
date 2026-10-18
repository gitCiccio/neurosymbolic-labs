# NumPy per capire e costruire una rete neurale

## Che cos'è

NumPy è la libreria fondamentale di Python per il calcolo numerico con array multidimensionali (`ndarray`). Un array contiene valori di un tipo omogeneo ed espone operazioni su interi vettori o matrici. In una rete neurale questi array possono rappresentare esempi, pesi, bias, attivazioni e gradienti.

NumPy **non costruisce automaticamente il grafo delle operazioni e non calcola da solo i gradienti**. È quindi molto utile per capire e implementare a mano forward pass, loss, backpropagation e aggiornamento dei pesi. Nel notebook del laboratorio 1 il confronto è tra un motore scalare scritto a mano e PyTorch; NumPy è un possibile passo intermedio, più efficiente dei cicli Python per i calcoli su matrici.

## La convenzione delle forme

Supponiamo di avere `B` esempi, `D` caratteristiche in ingresso e `H` neuroni in uno strato:

| Quantità | Forma | Significato |
|---|---:|---|
| `X` | `(B, D)` | Una riga per esempio |
| `W` | `(D, H)` | Un peso per collegamento |
| `b` | `(H,)` | Un bias per neurone |
| `Z = X @ W + b` | `(B, H)` | Somme pesate prima dell'attivazione |

`@` esegue il prodotto matriciale. Nell'addizione, NumPy applica il *broadcasting*: lo stesso vettore `b` viene sommato a ciascuna delle `B` righe. Il prodotto `X * W`, invece, sarebbe elemento per elemento e qui non sostituisce `X @ W`.

## Funzioni principali, ordinate per scopo

| Scopo | Funzioni e proprietà | Uso nella rete |
|---|---|---|
| Creare dati | `np.array`, `np.asarray`, `np.zeros`, `np.ones` | Input, etichette, gradienti iniziali |
| Inizializzare i pesi | `np.random.default_rng`, `rng.normal`, `rng.uniform` | Valori iniziali riproducibili con un seed |
| Controllare le dimensioni | `.shape`, `.ndim`, `.dtype`, `.reshape`, `.T` | Evitare prodotti e confronti tra forme sbagliate |
| Calcolare uno strato | `@`, `np.matmul`, `np.sum` | Somme pesate e propagazione dei gradienti |
| Attivare i neuroni | `np.maximum`, `np.exp`, `np.tanh` | ReLU, sigmoide e tanh |
| Misurare la perdita | `np.mean`, `np.square`, `np.log`, `np.clip` | MSE o entropia incrociata binaria |
| Gestire i dati | Indicizzazione, slicing, `rng.permutation` | Suddivisione e minibatch |
| Classificare | `>=`, `.astype`, `np.mean` | Soglia 0,5 e accuratezza |

Esempio minimo di creazione e forward pass:

```python
import numpy as np

rng = np.random.default_rng(42)
X = np.array([[0., 1.], [1., 0.]], dtype=np.float64)  # (2, 2)
W = rng.normal(0, 0.5, size=(2, 1))                  # (2, 1)
b = np.zeros(1)                                       # (1,)

z = X @ W + b                                         # (2, 1)
y_hat = 1.0 / (1.0 + np.exp(-z))                     # (2, 1)
```

Per una rete `D → H → 1`, il forward pass diventa:

```python
Z1 = X @ W1 + b1
A1 = np.maximum(Z1, 0.0)  # ReLU
Z2 = A1 @ W2 + b2
P = 1.0 / (1.0 + np.exp(-Z2))  # probabilità
```

La ReLU serve a introdurre non linearità. Senza un'attivazione non lineare tra due strati, i due prodotti matriciali si potrebbero combinare in un unico strato lineare.

## Loss, derivate e aggiornamento

Per classificazione binaria, con etichette `Y` di forma `(B, 1)`, si può usare l'entropia incrociata binaria:

```python
eps = 1e-12
P_safe = np.clip(P, eps, 1.0 - eps)
loss = -np.mean(Y * np.log(P_safe) + (1.0 - Y) * np.log(1.0 - P_safe))
```

Per una sigmoide finale combinata con questa loss, la derivata rispetto a `Z2` si semplifica. Se `B = len(X)`:

```python
dZ2 = (P - Y) / B
dW2 = A1.T @ dZ2
db2 = np.sum(dZ2, axis=0)

dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)  # derivata della ReLU
dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0)

W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
```

Questa è la regola della catena del laboratorio scritta con matrici. Prima si calcola l'effetto dell'errore sull'ultimo strato, poi si risale al primo. È importante calcolare **tutti** i gradienti prima di modificare un peso: le derivate di uno stesso passaggio devono riferirsi agli stessi parametri.

## Dati, batch e valutazione

```python
indices = rng.permutation(len(X))
train_idx = indices[:int(0.8 * len(X))]
test_idx = indices[int(0.8 * len(X)):]
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

batch_size = 32
for start in range(0, len(X_train), batch_size):
    xb = X_train[start:start + batch_size]
    yb = Y_train[start:start + batch_size]
    # forward → loss → derivate → aggiornamento
```

Per la valutazione di probabilità binarie: `pred = (P >= 0.5).astype(int)` e `accuracy = np.mean(pred == Y)`. Il test deve usare dati non impiegati per aggiornare i pesi.

## Collegamento al laboratorio 1

Il dataset del notebook contiene tutte le `2**10 = 1024` stringhe binarie di 10 bit. L'etichetta è `sum(bits) % 2`: vale 1 se il numero di bit uguali a 1 è dispari. È una generalizzazione della XOR a due bit. Con NumPy si possono rappresentare tutti gli esempi come `X.shape == (1024, 10)` e le etichette come `Y.shape == (1024, 1)`.

La parità a 10 bit è più difficile della XOR a due bit. Il fatto che una rete abbia abbastanza parametri non garantisce che una specifica inizializzazione e un certo numero di epoche la addestrino bene. Occorre misurare la prestazione sul test e confrontare configurazioni con le stesse suddivisioni dei dati.

## Riferimenti ufficiali

- [Introduzione a NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Creazione degli array](https://numpy.org/doc/stable/user/basics.creation.html)
- [Quickstart: prodotto matriciale e broadcasting](https://numpy.org/doc/stable/user/quickstart.html)

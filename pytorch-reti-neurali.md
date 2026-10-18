# PyTorch per creare e addestrare una rete neurale

## Che cos'è

PyTorch è una libreria per il calcolo su tensori e il machine learning. Un `torch.Tensor` somiglia a un array NumPy, ma può partecipare al calcolo automatico delle derivate (*autograd*) e può essere spostato su dispositivi supportati, per esempio una GPU. `torch.nn` fornisce strati e funzioni di perdita; `torch.optim` gestisce gli aggiornamenti dei parametri; `torch.utils.data` aiuta a preparare batch.

Nel laboratorio 1 PyTorch serve a confrontare un grafo di calcolo e una backpropagation scritti a mano con le versioni automatiche. La matematica rimane la stessa: PyTorch registra le operazioni durante il forward e usa la regola della catena durante `backward()`.

## Gli elementi principali

| Scopo | API | Significato |
|---|---|---|
| Creare dati | `torch.tensor`, `torch.zeros`, `torch.randn` | Tensori di input, etichette e parametri |
| Controllare le forme | `.shape`, `.reshape`, `.view`, `.unsqueeze` | Allineare batch, output ed etichette |
| Costruire la rete | `nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.Sequential` | Strati e parametri registrati automaticamente |
| Calcolare derivate | `requires_grad`, `loss.backward()`, `.grad` | Grafo dinamico e gradienti |
| Calcolare la loss | `nn.MSELoss`, `nn.BCELoss`, `nn.BCEWithLogitsLoss` | Errore da minimizzare |
| Aggiornare i pesi | `torch.optim.SGD`, `torch.optim.Adam`, `optimizer.step()` | Ottimizzazione |
| Organizzare i dati | `TensorDataset`, `DataLoader`, `random_split` | Dataset, batch e suddivisione |
| Valutare | `model.eval()`, `torch.no_grad()` | Inferenza senza registrare gradienti |

## Forme dei tensori

Per `B` esempi con `D` caratteristiche, `X` ha forma `(B, D)`. Con `nn.Linear(D, H)`, l'output ha forma `(B, H)`. Per una classificazione binaria con un output, i logit e le etichette dovrebbero avere entrambi forma `(B, 1)`. Una differenza tra `(B,)` e `(B, 1)` è una causa frequente di errori o confronti non desiderati.

## Un modello completo, piccolo e leggibile

Qui usiamo la XOR a due bit per mostrare il ciclo. Il dataset del laboratorio estende l'idea a **10 bit**.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])
loader = DataLoader(TensorDataset(X, Y), batch_size=4, shuffle=True)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(1000):
    for xb, yb in loader:
        optimizer.zero_grad()       # azzera i gradienti del passo precedente
        logits = model(xb)          # forward: (B, 1)
        loss = loss_fn(logits, yb)  # misura l'errore
        loss.backward()             # backpropagation
        optimizer.step()            # aggiorna pesi e bias

model.eval()
with torch.no_grad():
    probabilities = torch.sigmoid(model(X))
    predictions = (probabilities >= 0.5).float()
    accuracy = (predictions == Y).float().mean().item()
```

`BCEWithLogitsLoss` include internamente la sigmoide e lavora sui **logit**, cioè sulle uscite grezze dello strato finale. Per questo il modello dell'esempio non ha `nn.Sigmoid()` alla fine; la sigmoide compare soltanto quando vogliamo leggere le probabilità. Questa combinazione è più stabile numericamente di `Sigmoid` seguita da `BCELoss`.

Il piccolo esempio mostra la struttura del programma, non una garanzia di convergenza per ogni inizializzazione. Per misurare la generalizzazione bisogna inoltre valutare dati tenuti fuori dall'addestramento.

## Che cosa succede nel grafo del laboratorio

Per un singolo neurone con due input si calcola, in sostanza, `p = sigmoid(x @ w.T + b)`. La figura `graph.png` del laboratorio mostra i parametri `weight (1, 2)` e `bias (1)`, l'operazione lineare (`AddmmBackward0`) e l'attivazione (`SigmoidBackward0`). I nodi con suffisso `Backward0` rappresentano le regole di derivazione associate alle operazioni. `AccumulateGrad` raccoglie i gradienti nei parametri foglia. La figura rappresenta l'output del neurone; nell'addestramento il backward parte normalmente da una **loss** calcolata confrontando output ed etichetta.

Si può osservare il meccanismo con un tensore semplice:

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.sigmoid(x)
y.backward()
print(x.grad)  # dy/dx = sigmoid(x) * (1 - sigmoid(x))
```

I gradienti dei parametri si accumulano tra chiamate a `backward()`. Nel ciclo di addestramento `optimizer.zero_grad()` li azzera prima del nuovo passo.

## Applicazione alla parità a 10 bit

Nel notebook ogni input è una sequenza di 10 valori `0/1`, e l'etichetta è 1 se la somma dei bit è dispari. Una rete di partenza può avere `nn.Linear(10, 18)`, `nn.ReLU()` e `nn.Linear(18, 1)`. Per addestrarla: suddividi train/test, crea un `DataLoader` per il train, ripeti forward → loss → backward → step, poi misura loss e accuratezza sul test.

Per confrontare due implementazioni in modo attendibile usa gli stessi esempi di train/test, la stessa architettura, la stessa loss, lo stesso ottimizzatore e seed controllati. Nel notebook la versione manuale usa MSE e aggiornamenti espliciti, mentre quella PyTorch usa BCE e Adam: i risultati non sono quindi un confronto isolato della sola libreria.

## Riferimenti ufficiali

- [Introduzione a `torch.nn`](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)
- [Ciclo di addestramento PyTorch](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [Dataset e DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

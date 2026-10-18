# Neurosymbolic AI — Laboratori

Raccolta di implementazioni e appunti sviluppati durante lo studio dei laboratori del corso di Neurosymbolic AI.

## Lab 1 — Reti neurali e backpropagation

Il primo laboratorio approfondisce come una rete neurale calcola una previsione e apprende attraverso la discesa del gradiente. Il percorso svolto comprende una prima implementazione manuale in NumPy e la sua traduzione in PyTorch.

### Percorso svolto

1. **Singolo neurone in NumPy:** creazione di input, pesi casuali e bias; forward pass con somma pesata `X @ W + b` e sigmoide.
2. **Calcolo della loss:** introduzione delle etichette corrette e della Mean Squared Error (MSE).
3. **Backpropagation manuale:** applicazione della regola della catena per calcolare i gradienti dei pesi e del bias.
4. **Aggiornamento dei parametri:** discesa del gradiente e ripetizione del ciclo di addestramento per più epoche.
5. **XOR completa:** generazione dei quattro input binari con due cicli e valutazione delle previsioni. Studio del limite di un singolo neurone, che non può separare correttamente tutti i casi XOR.
6. **MLP in NumPy:** aggiunta di uno strato nascosto con ReLU e implementazione delle derivate per entrambi gli strati.
7. **MLP in PyTorch:** stessa architettura, con `nn.Module`, `nn.Linear`, autograd e ottimizzatore SGD.

### Problema XOR

La risposta corretta vale `1` quando i due bit sono diversi:

| Input | Etichetta |
|---|---:|
| `[0, 0]` | 0 |
| `[0, 1]` | 1 |
| `[1, 0]` | 1 |
| `[1, 1]` | 0 |

Gli input hanno forma `(4, 2)` e le etichette `(4, 1)`: ogni riga rappresenta un esempio.

### Architettura e addestramento del MLP

```text
2 ingressi → 8 neuroni nascosti con ReLU → 1 uscita con sigmoide
```

Il forward pass applica:

```text
z_hidden = X @ W1 + b1
hidden   = ReLU(z_hidden)
z_output = hidden @ W2 + b2
y_hat    = sigmoid(z_output)
```

L'addestramento usa MSE, learning rate `0.1`, 5.000 epoche e tutti e quattro gli esempi insieme a ogni aggiornamento. Le uscite vengono classificate con soglia `0.5`.

In NumPy i gradienti e gli aggiornamenti sono espliciti. In PyTorch il ciclo usa `optimizer.zero_grad()`, forward, loss, `loss.backward()` e `optimizer.step()`. SGD senza momentum applica lo stesso aggiornamento di discesa del gradiente della versione manuale.

### Risultati osservati

Entrambe le implementazioni MLP hanno raggiunto il **100% di accuratezza sui quattro esempi XOR usati per l'addestramento**. Nella verifica della versione NumPy la MSE è scesa da circa `0.26678` a `0.00140`; nell'esecuzione PyTorch riportata durante lo studio, la MSE finale è stata circa `0.00077`.

Questi risultati verificano la soluzione della tabella XOR completa a due bit. Non rappresentano una valutazione su un test set separato. I seed e le inizializzazioni di NumPy e PyTorch non producono necessariamente gli stessi parametri iniziali.

### File disponibili

| File | Contenuto |
|---|---|
| [NeuralNetworkMLPXor.py](neural_networks/NeuralNetworkMLPXor.py) | MLP NumPy con forward, backpropagation e aggiornamenti manuali |
| [NeuralNetworkXorPytorch.py](neural_networks/NeuralNetworkXorPytorch.py) | MLP equivalente in PyTorch con autograd e SGD |
| [Guida NumPy](docs/numpy-reti-neurali.md) | Array, forme, operazioni e derivate per una rete neurale |
| [Guida PyTorch](docs/pytorch-reti-neurali.md) | Tensori, modelli, autograd e ciclo di addestramento |

### Esecuzione

Con un interprete Python e un ambiente virtuale attivo, installare le dipendenze:

```powershell
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Dalla cartella principale del progetto:

```powershell
python neural_networks/NeuralNetworkMLPXor.py
python neural_networks/NeuralNetworkXorPytorch.py
```

Gli script mostrano l'andamento della loss, le previsioni finali e l'accuratezza sui quattro esempi.

### Parti del laboratorio ancora da affrontare

- Costruzione di un piccolo motore di differenziazione automatica con nodi di un grafo computazionale, come nel notebook del corso.
- Estensione alla parità a **10 bit**, con tutte le `2**10 = 1024` combinazioni.
- Suddivisione dei dati in training e test set.
- Esperimenti con il MLP a 18 neuroni nascosti e con una rete più profonda, previsti nella parte finale del notebook.

### Riferimenti

- Notebook del corso: `lab1-bp-nn.ipynb`.
- [NumPy — Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [PyTorch — Learning PyTorch with Examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)

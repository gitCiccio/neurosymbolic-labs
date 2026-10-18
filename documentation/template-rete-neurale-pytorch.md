# Come costruire una rete neurale con PyTorch

Guida al template di una rete MLP per classificazione binaria. L'esempio riprende l'esercizio sui tre bit: la risposta vale **0 se sono tutti uguali**, altrimenti **1**. Il template usa uno strato nascosto con ReLU, un'uscita con sigmoide, MSE e discesa del gradiente semplice, come negli esercizi svolti.

## 1. Prima del codice: definire il problema

Prima di costruire una rete, rispondi a queste domande:

1. Quali valori descrivono un esempio? Qui sono tre bit.
2. Qual è la risposta corretta? Un'etichetta binaria, 0 oppure 1.
3. Quanti esempi abbiamo? Le otto combinazioni dei tre bit.
4. Quale struttura scegliamo? Tre ingressi, otto neuroni nascosti e un'uscita.
5. Come misuriamo l'errore? Con la MSE, per continuità con il laboratorio.
6. Su quali dati misuriamo le prestazioni? In questo esercizio sugli stessi otto esempi di addestramento.

Il numero di esempi e il numero di neuroni nascosti sono due quantità indipendenti. Qui valgono entrambi otto per una coincidenza della configurazione scelta.

## 2. Capire i tensori e le forme

Un tensore contiene numeri organizzati lungo zero, una o più dimensioni. Uno scalare ha zero dimensioni; un vettore una; una matrice due. Gli input di questo esempio sono una matrice: una riga per esempio, una colonna per caratteristica.

| Oggetto | Forma | Significato |
|---|---|---|
| `X` | `(8, 3)` | Otto esempi, ciascuno con tre ingressi |
| `Y` | `(8, 1)` | Una risposta corretta per esempio |
| `z_hidden` | `(8, 8)` | Otto somme pesate per esempio |
| `hidden` | `(8, 8)` | Otto attivazioni dopo ReLU per esempio |
| `z_output` | `(8, 1)` | Una somma pesata finale per esempio |
| `y_hat` | `(8, 1)` | Una previsione per esempio |
| `loss` | `()` | Un tensore scalare che riassume l'errore |

Con `torch.tensor([[1.], [0.]])` crei due righe con una colonna, forma `(2, 1)`. Con `torch.tensor([1., 0.])` crei un vettore di forma `(2,)`. Le due strutture non sono intercambiabili quando confronti previsioni ed etichette: il broadcasting può produrre confronti indesiderati. Per questo controlliamo che `y_hat.shape == Y.shape`.

## 3. Il flusso completo

```text
PREPARAZIONE
    Crea X e Y
    Crea il modello: pesi e bias vengono inizializzati
    Crea la funzione di loss e l'ottimizzatore

ADDESTRAMENTO: ripeti per ogni epoca
    Azzera i gradienti precedenti
    X -> strato lineare -> ReLU -> strato lineare -> sigmoide -> y_hat
    Confronta y_hat con Y -> loss
    Percorri il grafo a ritroso -> gradienti
    Aggiorna pesi e bias

VALUTAZIONE: dopo il ciclo
    Stesso X -> modello aggiornato -> nuove previsioni
    Calcola loss finale, classi predette e accuratezza
```

Durante l'addestramento gli input e le etichette restano invariati. Cambiano i parametri della rete e, di conseguenza, le previsioni. `optimizer.step()` modifica i parametri; la loss già calcolata mantiene il suo valore fino a quando non ne calcoliamo una nuova.

## 4. Template completo

Salva questo blocco in un file Python, per esempio `NeuralNetworkThreeBits.py`, ed eseguilo con l'interprete in cui hai installato PyTorch. Il codice usa la CPU.

```python
import logging

import torch
from torch import nn


log = logging.getLogger(__name__)


def create_input():
    rows = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                rows.append([float(i), float(j), float(k)])
    return torch.tensor(rows, dtype=torch.float32)


def create_targets(X):
    targets = []
    for row in X:
        all_equal = bool(torch.all(row == row[0]).item())
        targets.append([0.0 if all_equal else 1.0])
    return torch.tensor(targets, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, in_features, n_hidden, n_outputs):
        super().__init__()
        self.hidden_layer = nn.Linear(in_features, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_outputs)

    def forward(self, X):
        z_hidden = self.hidden_layer(X)
        hidden = torch.relu(z_hidden)
        z_output = self.output_layer(hidden)
        y_hat = torch.sigmoid(z_output)
        return y_hat


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    torch.manual_seed(42)

    X = create_input()
    Y = create_targets(X)
    model = MLP(in_features=X.shape[1], n_hidden=8, n_outputs=1)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 5000

    log.info("Modello:\n%s", model)
    log.info("Forme: X=%s, Y=%s", tuple(X.shape), tuple(Y.shape))
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_hat = model(X)
        assert y_hat.shape == Y.shape
        loss = loss_fn(y_hat, Y)

        if epoch == 1:
            loss_initial = loss.item()

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 500 == 0:
            log.info("Epoca %d | Loss prima del passo: %.8f", epoch, loss.item())

    # Questo blocco deve essere fuori dal ciclo delle epoche.
    model.eval()
    with torch.no_grad():
        y_hat_final = model(X)
        loss_final = loss_fn(y_hat_final, Y).item()
        predictions = (y_hat_final >= 0.5).to(torch.int64)
        accuracy = (predictions == Y).float().mean().item()

    for x, target, probability, prediction in zip(X, Y, y_hat_final, predictions):
        log.info(
            "Input: %s | Atteso: %d | Output: %.4f | Predetto: %d",
            x.tolist(), int(target.item()), probability.item(), prediction.item(),
        )

    log.info("Loss iniziale: %.8f | Loss finale: %.8f", loss_initial, loss_final)
    log.info("Accuratezza sugli otto esempi di training: %.2f%%", accuracy * 100)


if __name__ == "__main__":
    main()
```

## 5. Preparazione: cosa fanno le funzioni

### Import e logging

`import torch` rende disponibili tensori e operazioni. `from torch import nn` importa gli strumenti per gli strati e le loss. `logging.getLogger(__name__)` recupera il logger del modulo; `logging.basicConfig(level=logging.INFO, ...)` abilita i messaggi `INFO` nella console. Il solo `getLogger` non abilita quel livello.

### Dati ed etichette

`create_input()` genera le otto combinazioni con tre cicli. Ogni iterazione aggiunge una riga di tre valori. `torch.tensor(..., dtype=torch.float32)` converte le liste in un tensore con numeri in virgola mobile a 32 bit, lo stesso tipo usato normalmente dai parametri del modello appena creato.

In `create_targets(X)`, `row == row[0]` confronta tutti i valori della riga con il primo. `torch.all(...)` controlla che tutti i confronti siano veri. `.item()` estrae il risultato booleano scalare. Se i bit sono tutti uguali aggiungiamo `[0.0]`, altrimenti `[1.0]`. Le parentesi interne mantengono una colonna in `Y`.

I tuoi cicli di confronto tra elementi adiacenti sono un'alternativa corretta a questa implementazione. Le etichette sono create dalla regola nota del problema: durante il forward il modello riceve soltanto `X`.

### Seed

`torch.manual_seed(42)` inizializza i generatori casuali di PyTorch. Va chiamato prima della costruzione del modello, perché `nn.Linear` inizializza i parametri quando viene creato. Il seed aiuta a ripetere l'esperimento nello stesso ambiente; non garantisce risultati identici tra librerie, dispositivi o versioni differenti.

## 6. Modello: struttura e forward

`class MLP(nn.Module)` definisce un modello PyTorch. `__init__` descrive gli strati disponibili; `super().__init__()` inizializza la parte ereditata da `nn.Module`. Assegnando gli strati a `self.hidden_layer` e `self.output_layer`, PyTorch registra anche i loro parametri.

`nn.Linear(3, 8)` crea otto neuroni che ricevono tre valori ciascuno. `nn.Linear(8, 1)` crea un neurone finale che riceve le otto attivazioni nascoste. L'uscita del primo strato deve quindi avere lo stesso numero di caratteristiche atteso dall'ingresso del secondo.

Attenzione alla convenzione dei pesi: `nn.Linear(in_features, out_features)` conserva `weight` con forma `(out_features, in_features)` e calcola `X @ weight.T + bias`. Nel codice NumPy avevamo conservato direttamente la matrice nella forma opposta. Per questo i pesi PyTorch dei due strati hanno forme `(8, 3)` e `(1, 8)`.

`model(X)` esegue il percorso definito in `forward`. Ogni somma pesata è seguita da un'attivazione:

- `torch.relu(z)` sostituisce i valori negativi con zero e mantiene quelli positivi. Introduce non linearità senza cambiare la forma.
- `torch.sigmoid(z)` calcola `1 / (1 + exp(-z))`, producendo un'uscita tra 0 e 1 per ciascun esempio.

La parola *hidden layer* indica lo strato nascosto. `hidden` contiene le sue attivazioni per il batch corrente: in questo MLP non è una memoria conservata tra esempi successivi.

## 7. Loss, backpropagation e aggiornamento

| Istruzione | Effetto | Cosa rimane invariato |
|---|---|---|
| `loss_fn = nn.MSELoss()` | Crea l'oggetto che calcolerà la MSE | Non calcola ancora una loss |
| `optimizer = torch.optim.SGD(...)` | Prepara l'ottimizzatore e i parametri da aggiornare | I parametri non sono ancora aggiornati |
| `optimizer.zero_grad()` | Azzera o rimuove i gradienti precedenti | Pesi e bias conservano i valori appresi |
| `y_hat = model(X)` | Esegue il forward con i parametri attuali | Gli input e i parametri non vengono aggiornati |
| `loss = loss_fn(y_hat, Y)` | Calcola la media degli errori quadratici | Non aggiorna i parametri |
| `loss.backward()` | Calcola e accumula i gradienti nei parametri | Non modifica i valori dei parametri |
| `optimizer.step()` | Aggiorna i parametri usando i gradienti | Non ricalcola previsioni o loss |

La MSE è `mean((y_hat - Y) ** 2)`. La loss viene calcolata fuori dal modello perché richiede le risposte corrette, mentre il modello deve poter prevedere anche quando quelle risposte non sono disponibili.

I parametri di `nn.Linear` hanno `requires_grad=True` automaticamente. Le operazioni che li usano vengono registrate da autograd. Perciò `loss.backward()` può applicare la regola della catena e salvare, per esempio, `model.hidden_layer.weight.grad`. Non è necessario impostare `requires_grad=True` sugli input per addestrare i pesi in questo esercizio.

PyTorch accumula i gradienti tra chiamate a `backward()`. Nel nostro ciclo li azzeriamo prima di calcolarne di nuovi. SGD senza momentum e senza altre opzioni applica a ogni parametro `p` l'aggiornamento `p = p - lr * grad`.

`model.parameters()` fornisce i parametri registrati nel modello. La rete ha 41 parametri: `3*8 + 8` nel primo strato e `8*1 + 1` nel secondo.

## 8. Epoche e loss stampata

Un'epoca è un passaggio completo sul training set. Qui usiamo tutti gli otto esempi insieme: ogni epoca coincide con un unico aggiornamento. Se dividessimo i dati in minibatch, ogni epoca potrebbe contenere più aggiornamenti.

`loss.item()` converte un tensore con un solo elemento in un valore Python. Salviamo così la loss della prima epoca e stampiamo le altre. Scrivere `loss.item` senza parentesi restituisce il metodo, non il numero, e provoca un errore se lo passi al formato `%.8f`.

Il valore stampato nel ciclo è la loss calcolata prima dell'aggiornamento di quell'epoca. Per sapere la loss dopo l'ultimo aggiornamento eseguiamo un nuovo forward fuori dal ciclo.

## 9. Valutazione e conversione in classi

`model.train()` e `model.eval()` impostano rispettivamente le modalità di addestramento e valutazione. Non eseguono da sole né addestramento né previsioni. Cambiano il comportamento di alcuni strati, come Dropout e BatchNorm; con Linear, ReLU e sigmoide il calcolo rimane uguale.

`with torch.no_grad():` disabilita la registrazione delle operazioni per le derivate nel blocco. È distinto da `eval()`. Le variabili assegnate nel blocco rimangono accessibili anche dopo. La valutazione finale deve essere fuori dal ciclo delle epoche; per valutazioni intermedie occorre gestire esplicitamente il ritorno a `model.train()`.

Il confronto `y_hat_final >= 0.5` produce booleani. `.to(torch.int64)` li converte nelle classi 0 e 1. Il confronto `predictions == Y` indica quali risposte sono corrette; `.float()` converte vero e falso in 1.0 e 0.0; `.mean().item()` restituisce la frazione di risposte corrette.

`zip(X, Y, y_hat_final, predictions)` prende una riga da ciascuna struttura nella stessa posizione. Le quattro variabili del `for` ricevono input, etichetta, output e classe dello stesso esempio. `zip` termina quando finisce la struttura più corta; qui tutte hanno otto righe. `.tolist()` converte una riga in una lista Python per stamparla; `.item()` estrae il singolo numero dalle altre righe.

## 10. Come adattare il template

| Cosa cambia | Dove intervenire |
|---|---|
| Numero di caratteristiche | Costruzione di `X`; `in_features=X.shape[1]` lo legge automaticamente |
| Numero di esempi | Numero di righe di `X` e `Y`, senza cambiare automaticamente i neuroni |
| Regola da apprendere | Generazione delle etichette in `create_targets` |
| Capacità dello strato nascosto | `n_hidden`, scelto indipendentemente dagli esempi |
| Velocità dei passi | `lr` nell'ottimizzatore |
| Numero di passaggi sui dati | `epochs` |
| Tipo di problema | Uscita del modello, tipo e forma delle etichette, loss e metrica |

Per XOR a due bit, genera due bit per riga e assegna 1 quando sono diversi. Per la parità a dieci bit, genera dieci bit e usa come etichetta `sum(bits) % 2`. L'uscita resta una sola per esempio, ma architettura e addestramento possono richiedere nuovi esperimenti.

Il template mantiene sigmoide + MSE per collegarsi alle derivate già studiate. Per classificazione binaria è comune usare `nn.BCEWithLogitsLoss`: in quel caso il modello deve restituire `z_output` senza sigmoide, perché la loss la incorpora in modo numericamente stabile. Applica la sigmoide successivamente quando vuoi leggere le probabilità. Per regressione o classificazione con più classi servono altre combinazioni: non basta aumentare `n_outputs` lasciando invariato tutto il resto.

## 11. Interpretare i risultati e controllare gli errori

- La loss misura l'errore continuo degli output; l'accuratezza conta le decisioni corrette dopo la soglia. Non devono necessariamente migliorare insieme a ogni passo.
- Un'accuratezza del 100% sugli otto esempi verifica la tabella completa di questo piccolo problema. Per valutare la generalizzazione su nuovi esempi in problemi più grandi serve un test set separato; per scegliere configurazioni è utile anche un validation set.
- Le sei etichette 1 e le due etichette 0 rendono il dataset sbilanciato: predire sempre 1 darebbe già il 75%. Controlla anche le singole risposte.
- Seed, learning rate e 5.000 epoche non garantiscono la convergenza per qualsiasi architettura o problema.
- Se compare un errore su `.shape`, verifica che la variabile contenga davvero un tensore e che l'ordine dei valori restituiti e assegnati coincida.
- Se fallisce `matmul`, controlla le forme: l'ultimo numero della matrice a sinistra deve coincidere con il penultimo di quella a destra.
- Se passi molti argomenti a una funzione, i nomi espliciti aiutano a evitare scambi tra pesi e attivazioni.
- `if __name__ == "__main__": main()` avvia il programma quando esegui direttamente il file; importarlo rende disponibili le definizioni senza lanciare il training.

## Riferimenti ufficiali

- [Costruire un modello con PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [nn.Linear: convenzione di pesi e bias](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Autograd e gradienti](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)
- [Azzeramento dei gradienti](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)
- [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

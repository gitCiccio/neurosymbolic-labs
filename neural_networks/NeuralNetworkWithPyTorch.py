"""XOR con PyTorch: MLP 2 -> 8 -> 1, MSE e gradient descent."""

import logging
import torch
from torch import nn

log = logging.getLogger(__name__)

def create_input():
    input_x_array = []
    for i in range(2):
        for j in range(2):
            input_x_array.append((float(i), float(j)))
    return torch.tensor(input_x_array, dtype=torch.float32)

def create_Y():
    return torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

class XorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear crea e registra automaticamente pesi e bias
        self.hidden_layer = nn.Linear(in_features=2, out_features=8)
        self.output_layer = nn.Linear(in_features=8, out_features=1)

    def forward(self, X):
        z_hidden = self.hidden_layer(X)
        hidden = torch.relu(z_hidden)
        z_output = self.output_layer(hidden)
        y_hat = torch.sigmoid(z_output)
        return y_hat

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    torch.manual_seed(42)

    X = create_input()
    Y = create_Y()
    model = XorMLP()
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 5000

    log.info("Modello:\n%s", model)
    log.info("Forme: X=%s, Y=%s", tuple(X.shape), tuple(Y.shape))
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        y_hat = model(X)
        loss = loss_fn(y_hat, Y)
        if epoch == 1:
            loss_initial = loss.item()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 500 == 0:
            log.info("Epoca %d | Loss prima del passo: %.8f", epoch, loss.item())

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
        log.info("Accuratezza sui quattro esempi di training: %.2f%%", accuracy * 100)

if __name__ == "__main__":
    main()
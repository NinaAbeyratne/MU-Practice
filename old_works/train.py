import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")
x = torch.randn(2, 3).to(device)

print("Torch version:", torch.__version__)
print("Device:", device)
print("Tensor:", x)

def train(model, loader, epochs, save_epoch=1):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        if epoch == save_epoch:
            torch.save(model.state_dict(), "models/early_checkpoint.pt")

    torch.save(model.state_dict(), "models/full_model.pt")

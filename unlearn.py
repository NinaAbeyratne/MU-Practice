device = torch.device("cpu")
model.to(device)
x = torch.randn(2, 3).to(device)

# unlearn.py
import torch
import torch.nn as nn
import torch.optim as optim

def first_epoch_reversal(model, retain_loader, epochs=3):
    model.load_state_dict(torch.load("models/early_checkpoint.pt"))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in retain_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "models/reversal_unlearned.pt")
    return model

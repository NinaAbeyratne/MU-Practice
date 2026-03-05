device = torch.device("cpu")
model.to(device)
x = torch.randn(2, 3).to(device)
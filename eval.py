import torch

def evaluate(model, dataloaders, device):
    model.eval()
    results = []
    with torch.no_grad():
        for loader in dataloaders:
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
            results.append(correct / total)
    return results

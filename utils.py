import torch
import torch.nn.functional as F

def prototype_alignment(model, dataloader, device):
    model.eval()
    class_features = {}

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            features = model.extract_features(x)

            for f, label in zip(features, y):
                label = label.item()
                class_features.setdefault(label, []).append(f)

    for cls, feats in class_features.items():
        feats = torch.stack(feats)
        proto = feats.mean(dim=0)
        proto = F.normalize(proto, dim=0)
        model.classifier.weight.data[cls] = proto

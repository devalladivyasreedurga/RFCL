from torchvision import datasets, transforms
from torch.utils.data import Subset

CLASSES_PER_TASK = 20

# ImageNet stats — required since backbone is pretrained on ImageNet
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def get_cifar100_tasks(num_tasks=5):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_data = datasets.CIFAR100(root='./data', train=True,  download=True, transform=train_transform)
    test_data  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    tasks = []
    for t in range(num_tasks):
        start = t * CLASSES_PER_TASK
        end   = (t + 1) * CLASSES_PER_TASK

        train_idx = [i for i, (_, y) in enumerate(train_data) if start <= y < end]
        test_idx  = [i for i, (_, y) in enumerate(test_data)  if start <= y < end]

        tasks.append((Subset(train_data, train_idx), Subset(test_data, test_idx)))

    return tasks

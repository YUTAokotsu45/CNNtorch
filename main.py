import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def main():
    # 1. Tous les cœurs du processeur
    device = torch.device("cpu")
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    print(f"Utilisation de {num_cores} cœurs CPU.")

    # 2. Image en 64x64 au lieu de 128x128 (4x plus rapide !)
    IMG_SIZE = 64
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    path = r"leaves_healthy_or_diseased"
    full_dataset = ImageFolder(path, transform=transform)
    classes = full_dataset.classes

    # Sélection de 5000 images (2500 par classe)
    random.seed(42)
    class_indices = {}
    for i, (_, label) in enumerate(full_dataset.samples):
        class_indices.setdefault(label, []).append(i)

    selected_indices = []
    for label, indices in class_indices.items():
        selected_indices.extend(random.sample(indices, min(2500, len(indices))))
    random.shuffle(selected_indices)

    # FAST LOADING PAR BATCHES (Évite d'ouvrir 5000 fois les fichiers un par un)
    print("Préchargement des images en RAM...")
    t0 = time.time()
    
    subset = torch.utils.data.Subset(full_dataset, selected_indices)
    loader_tmp = DataLoader(subset, batch_size=256, num_workers=0)

    x_list, y_list = [], []
    for imgs, lbls in loader_tmp:
        x_list.append(imgs)
        y_list.append(lbls)

    X_all = torch.cat(x_list, dim=0)
    Y_all = torch.cat(y_list, dim=0)
    print(f"Images préchargées en RAM en {time.time() - t0:.2f}s !")

    # Split (70% train / 10% val / 20% test)
    total_len = len(X_all)
    train_size = int(0.7 * total_len)
    val_size = int(0.1 * total_len)

    X_train, Y_train = X_all[:train_size], Y_all[:train_size]
    X_val, Y_val = X_all[train_size:train_size + val_size], Y_all[train_size:train_size + val_size]
    X_test, Y_test = X_all[train_size + val_size:], Y_all[train_size + val_size:]

    # DataLoader rapides
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=128, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=128, shuffle=False)

    # VOTRE CNN ADAPTÉ À 64x64
    model = nn.Sequential(
        # 64x64 -> 64x64
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2), # -> 32x32

        # 32x32 -> 32x32
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2), # -> 16x16

        # 16x16 -> 16x16
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2), # -> 8x8

        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Dropout(0.45),
        nn.Linear(64 * 4 * 4, 2)
    ).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)

    print("\nDébut du training rapide...")
    for epoch in range(30):
        t_epoch = time.time()
        model.train()
        correct, total, total_loss = 0, 0, 0.0

        for image, label in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(image)
            loss = loss_func(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == label).sum().item()
            total += label.size(0)

        train_acc = correct / total

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for image, label in val_loader:
                pred = model(image)
                val_correct += (pred.argmax(dim=1) == label).sum().item()
                val_total += label.size(0)

        val_acc = val_correct / val_total
        print(f"Époque {epoch+1:02d} ({time.time() - t_epoch:.2f}s) | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

if __name__ == "__main__":
    main()

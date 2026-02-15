# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from config import *
from dataset import SteelSurfaceDataset
from model import get_model
from utils import AverageMeter, save_checkpoint, accuracy

def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train() # Modo entrenamiento
    losses = AverageMeter()
    accs = AverageMeter()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Métricas
        acc = accuracy(outputs, labels)
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))
        
        if i % 10 == 0:
            print(f"Epoch [{epoch}][{i}/{len(train_loader)}] Loss: {losses.val:.4f} Acc: {accs.val:.2f}%")

def validate(val_loader, model, criterion):
    model.eval() # Modo evaluación (apaga Dropout, BatchNorm fija estadísticas)
    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad(): # No calculamos gradientes (ahorra memoria)
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            acc = accuracy(outputs, labels)
            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))

    print(f"--> VALIDATION Loss: {losses.avg:.4f} Acc: {accs.avg:.2f}%")
    return losses.avg

def main():
    # 1. Preparar Datos
    # Transformaciones: Normalización estándar de ImageNet
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Importante: Apunta a la carpeta 'processed' que crea el script prepare_data.py
    full_dataset = SteelSurfaceDataset(root_dir=os.path.join(DATA_DIR, "processed"), transform=transform)
    
    # Dividir Train/Val (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Modelo, Loss y Optimizador
    model = get_model(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Bucle Principal
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion)

        # Guardar si mejora
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

if __name__ == "__main__":
    main()
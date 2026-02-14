# src/utils.py
import torch
import os
import matplotlib.pyplot as plt
from config import MODELS_DIR

class AverageMeter:
    """Calcula y almacena el promedio y valor actual"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Guarda el modelo entrenado"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, filename)
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(MODELS_DIR, 'model_best.pth')
        torch.save(state, best_path)
        print(f"--> Guardado Nuevo Mejor Modelo: {best_path}")

def accuracy(output, target):
    """Calcula la precisión top-1"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()
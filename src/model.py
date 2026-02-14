# src/model.py
import torch.nn as nn
from torchvision import models

def get_model(num_classes: int, pretrained: bool = True):
    """
    Crea un modelo ResNet18 adaptado para nuestro número de clases.
    
    Args:
        num_classes (int): Número de categorías de defectos.
        pretrained (bool): Si es True, usa pesos de ImageNet (Transfer Learning).
    """
    # 1. Cargamos la arquitectura base (ResNet18 es ligera y potente)
    # 'weights="DEFAULT"' es la forma moderna de decir pretrained=True en versiones nuevas de Torch
    model = models.resnet18(weights='DEFAULT' if pretrained else None)
    
    # 2. Congelamos las capas base (Opcional: para no dañar los pesos aprendidos al inicio)
    # Para este ejercicio, dejaremos que todo se entrene (Fine-tuning completo) porque
    # las texturas de acero son muy diferentes a las de ImageNet (gatos/perros).
    
    # 3. Sustituimos la última capa (Fully Connected) para que coincida con nuestras clases
    # ResNet original tiene 1000 salidas, nosotros queremos 6.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
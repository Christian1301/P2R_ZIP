"""
Train Utils V15 - Utilities per training
Contiene collate_fn corretta e altre utility

CORREZIONE CRITICA:
- collate_fn arrotonda dimensioni a multipli di 16 per VGG backbone
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any


def _round_to_multiple(x: int, multiple: int = 16) -> int:
    """
    Arrotonda x al multiplo superiore di 'multiple'.
    CRITICO per VGG backbone che richiede dimensioni multiple di 16.
    """
    return ((x + multiple - 1) // multiple) * multiple


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function per DataLoader.
    
    CORREZIONE CRITICA: Arrotonda dimensioni a multipli di 16
    per evitare mismatch con VGG backbone.
    
    Args:
        batch: Lista di sample dict con 'image', 'points', 'count'
        
    Returns:
        Batch dict con tensori impilati e padding uniforme
    """
    images = []
    points_list = []
    counts = []
    original_sizes = []
    
    # Trova dimensioni massime
    max_h = max(item['image'].shape[1] for item in batch)
    max_w = max(item['image'].shape[2] for item in batch)
    
    # CORREZIONE: Arrotonda a multipli di 16
    max_h = _round_to_multiple(max_h, 16)
    max_w = _round_to_multiple(max_w, 16)
    
    for item in batch:
        img = item['image']
        h, w = img.shape[1], img.shape[2]
        
        # Salva dimensioni originali (per debug)
        original_sizes.append((h, w))
        
        # Padding (destra e basso)
        pad_w = max_w - w
        pad_h = max_h - h
        
        if pad_w > 0 or pad_h > 0:
            padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            padded = img
        
        images.append(padded)
        
        # Points (già tensori)
        if 'points' in item:
            points_list.append(item['points'])
        else:
            points_list.append(torch.empty(0, 2))
        
        # Count
        if 'count' in item:
            counts.append(item['count'])
        else:
            counts.append(len(item.get('points', [])))
    
    return {
        'image': torch.stack(images),
        'points': points_list,
        'count': torch.tensor(counts, dtype=torch.float32),
        'original_sizes': original_sizes
    }


def collate_fn_val(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function per validation (batch_size=1 tipicamente).
    Non fa padding per preservare dimensioni originali.
    """
    if len(batch) == 1:
        item = batch[0]
        img = item['image']
        h, w = img.shape[1], img.shape[2]
        
        # Arrotonda comunque a multipli di 16
        new_h = _round_to_multiple(h, 16)
        new_w = _round_to_multiple(w, 16)
        
        if new_h != h or new_w != w:
            img = F.pad(img, (0, new_w - w, 0, new_h - h), mode='constant', value=0)
        
        return {
            'image': img.unsqueeze(0),
            'points': [item.get('points', torch.empty(0, 2))],
            'count': torch.tensor([item.get('count', 0)], dtype=torch.float32),
            'original_size': (h, w)
        }
    else:
        return collate_fn(batch)


class AverageMeter:
    """Computa e memorizza media e valore corrente."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricTracker:
    """Traccia multiple metriche durante il training."""
    
    def __init__(self, metrics: List[str]):
        self.metrics = {name: AverageMeter(name) for name in metrics}
    
    def update(self, name: str, val: float, n: int = 1):
        if name in self.metrics:
            self.metrics[name].update(val, n)
    
    def reset(self):
        for meter in self.metrics.values():
            meter.reset()
    
    def get_avg(self, name: str) -> float:
        return self.metrics[name].avg if name in self.metrics else 0.0
    
    def get_all(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.metrics.items()}


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: str = None
):
    """
    Salva checkpoint del modello.
    
    Args:
        state: Dict con model_state_dict, optimizer_state_dict, epoch, metrics
        filepath: Path per salvare
        is_best: Se True, salva anche come best model
        best_filepath: Path per best model (opzionale)
    """
    import shutil
    
    torch.save(state, filepath)
    
    if is_best and best_filepath:
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None):
    """
    Carica checkpoint.
    
    Returns:
        Dict con info aggiuntive (epoch, metrics, etc.)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {k: v for k, v in checkpoint.items() 
            if k not in ['model_state_dict', 'optimizer_state_dict']}


def count_parameters(model: torch.nn.Module) -> int:
    """Conta parametri trainabili."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_module(module: torch.nn.Module):
    """Congela tutti i parametri di un modulo."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module):
    """Scongela tutti i parametri di un modulo."""
    for param in module.parameters():
        param.requires_grad = True


def set_bn_eval(module: torch.nn.Module):
    """
    Imposta BatchNorm in eval mode.
    Utile quando si vuole congelare BN durante fine-tuning.
    """
    for m in module.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.eval()


# Test
if __name__ == "__main__":
    # Test collate_fn
    batch = [
        {
            'image': torch.randn(3, 100, 150),
            'points': torch.tensor([[10, 20], [30, 40]]),
            'count': 2
        },
        {
            'image': torch.randn(3, 120, 130),
            'points': torch.tensor([[50, 60]]),
            'count': 1
        }
    ]
    
    collated = collate_fn(batch)
    
    print("=== Collate Function Test ===")
    print(f"Image batch shape: {collated['image'].shape}")
    print(f"Expected: multiple of 16 -> H={_round_to_multiple(120, 16)}, W={_round_to_multiple(150, 16)}")
    print(f"Actual: H={collated['image'].shape[2]}, W={collated['image'].shape[3]}")
    print(f"Points: {[p.shape for p in collated['points']]}")
    print(f"Counts: {collated['count']}")
    
    assert collated['image'].shape[2] % 16 == 0, "Height not multiple of 16!"
    assert collated['image'].shape[3] % 16 == 0, "Width not multiple of 16!"
    
    print("\n✅ train_utils V15 tests passed!")

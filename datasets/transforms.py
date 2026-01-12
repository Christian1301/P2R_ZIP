import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np

class TrainTransforms:
    """
    Gestisce le trasformazioni congiunte per Immagine, Punti e Densità.
    Supporta RandomResizedCrop e HorizontalFlip sincronizzati.
    """
    def __init__(self, cfg):
        self.mean = cfg.get("NORM_MEAN", [0.485, 0.456, 0.406])
        self.std = cfg.get("NORM_STD", [0.229, 0.224, 0.225])
        self.crop_size = cfg.get("CROP_SIZE", 384)
        
        # Configurazione Augmentation
        aug_cfg = cfg.get("AUGMENTATION", {})
        
        # Random Resized Crop (Logica nuova o fallback)
        self.rrc_enabled = False
        if "RandomResizedCrop" in aug_cfg:
            rrc = aug_cfg["RandomResizedCrop"]
            if rrc.get("ENABLED", False):
                self.rrc_enabled = True
                self.scale = rrc.get("SCALE", [0.08, 1.0])
                self.ratio = rrc.get("RATIO", [0.75, 1.33])
        # Fallback per config vecchi
        elif cfg.get("CROP_SCALE", [1.0, 1.0])[0] < 1.0:
            self.rrc_enabled = True
            self.scale = cfg.get("CROP_SCALE")
            self.ratio = (0.75, 1.33)

        # Horizontal Flip
        self.flip_prob = 0.5
        if "HorizontalFlip" in aug_cfg:
            if not aug_cfg["HorizontalFlip"].get("ENABLED", True):
                self.flip_prob = 0.0
            else:
                self.flip_prob = aug_cfg["HorizontalFlip"].get("PROB", 0.5)
        elif not cfg.get("HORIZONTAL_FLIP", True):
            self.flip_prob = 0.0

        # Augmentation Fotometriche (Solo Immagine)
        self.color_aug = None
        color_cfg = cfg.get("COLOR_JITTER", {})
        if color_cfg.get("ENABLED", False) or aug_cfg.get("COLOR_JITTER", {}).get("ENABLED", False):
            # Supporto per entrambi i formati config
            c = color_cfg if color_cfg else aug_cfg.get("COLOR_JITTER")
            self.color_aug = T.ColorJitter(
                brightness=c.get("BRIGHTNESS", 0.2),
                contrast=c.get("CONTRAST", 0.2),
                saturation=c.get("SATURATION", 0.2),
                hue=c.get("HUE", 0.1)
            )
            
        self.blur_aug = None
        blur_cfg = cfg.get("GAUSSIAN_BLUR", {})
        if blur_cfg.get("ENABLED", False):
            self.blur_aug = T.GaussianBlur(kernel_size=blur_cfg.get("KERNEL_SIZE", [3, 5]))
            self.blur_prob = blur_cfg.get("PROB", 0.1)

        self.gray_prob = cfg.get("RANDOM_GRAY_PROB", 0.0)

    def __call__(self, img, points, density=None):
        """
        img: PIL Image
        points: numpy array (N, 2) [x, y]
        density: numpy array (H, W) o None
        """
        # 1. Random Resized Crop (Sincronizzato)
        if self.rrc_enabled:
            # Calcola parametri random
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)
            
            # Applica a Immagine
            img = TF.resized_crop(img, i, j, h, w, (self.crop_size, self.crop_size))
            
            # Applica a Densità (se esiste)
            if density is not None:
                # La densità è un array numpy, va convertita in tensore per resized_crop o gestita
                # Qui assumiamo sia gestita come immagine 1-channel se necessario, ma spesso è None in P2R training
                # Se density è usata, andrebbe croppata. P2R rigenera la density dai punti, quindi:
                pass 

            # Applica ai Punti
            if len(points) > 0:
                # Trasla
                points[:, 0] -= j # x
                points[:, 1] -= i # y
                # Scala
                scale_w = self.crop_size / w
                scale_h = self.crop_size / h
                points[:, 0] *= scale_w
                points[:, 1] *= scale_h
                
                # Filtra punti fuori dal crop
                mask = (points[:, 0] >= 0) & (points[:, 0] < self.crop_size) & \
                       (points[:, 1] >= 0) & (points[:, 1] < self.crop_size)
                points = points[mask]
        else:
            # Fallback: Random Crop semplice (Fixed size)
            if img.size[0] < self.crop_size or img.size[1] < self.crop_size:
                # Padding se l'immagine è troppo piccola
                pad_w = max(0, self.crop_size - img.size[0])
                pad_h = max(0, self.crop_size - img.size[1])
                img = TF.pad(img, (0, 0, pad_w, pad_h))
                
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
            img = TF.crop(img, i, j, h, w)
            
            if len(points) > 0:
                points[:, 0] -= j
                points[:, 1] -= i
                mask = (points[:, 0] >= 0) & (points[:, 0] < self.crop_size) & \
                       (points[:, 1] >= 0) & (points[:, 1] < self.crop_size)
                points = points[mask]

        # 2. Random Horizontal Flip (Sincronizzato)
        if random.random() < self.flip_prob:
            img = TF.hflip(img)
            if len(points) > 0:
                # Flip coordinata x: W - x
                points[:, 0] = self.crop_size - points[:, 0]

        # 3. Augmentation Fotometriche (Solo Immagine)
        if self.color_aug:
            img = self.color_aug(img)
            
        if self.gray_prob > 0 and random.random() < self.gray_prob:
            img = TF.to_grayscale(img, num_output_channels=3)
            
        if self.blur_aug and random.random() < self.blur_prob:
            img = self.blur_aug(img)

        # 4. ToTensor e Normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)
        
        # Importante: BaseDataset si aspetta una tupla
        # points deve essere tensore
        points = torch.from_numpy(points).float()
        
        # Density di solito viene ricalcolata o è None, ritorniamo None se non gestita specificamente
        # Se il dataset richiede density trasformata, andrebbe fatto uno scaling della somma.
        # Per P2R standard, la density è generata dai punti on-the-fly o ignorata qui.
        
        return img, points, density

class ValTransforms:
    """Trasformazioni semplici per validazione (solo resize/norm)."""
    def __init__(self, cfg):
        self.mean = cfg.get("NORM_MEAN", [0.485, 0.456, 0.406])
        self.std = cfg.get("NORM_STD", [0.229, 0.224, 0.225])
        
    def __call__(self, img, points, density=None):
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)
        points = torch.from_numpy(points).float()
        return img, points, density

def build_transforms(cfg, is_train=True):
    if is_train:
        return TrainTransforms(cfg)
    else:
        return ValTransforms(cfg)
# P2R_ZIP/datasets/transforms.py
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2
from PIL import Image # Assicurati che PIL sia importato

# === CLASSI DI TRASFORMAZIONE ===

class Compose(object):
    """Applica una sequenza di trasformazioni."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None, den=None):
        for t in self.transforms:
            img, pts, den = t(img, pts, den)
        return img, pts, den

class ToTensor(object):
    """Converte immagine (PIL o Numpy) e densità (Numpy) in tensori."""
    def __call__(self, img, pts=None, den=None):
        if isinstance(img, Image.Image):
             img = F.to_tensor(img)
        elif isinstance(img, np.ndarray):
             img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        else:
             raise TypeError(f"Tipo immagine non supportato in ToTensor: {type(img)}")

        den = torch.from_numpy(den).unsqueeze(0) if den is not None else None
        return img, pts, den

class Normalize(object):
    """Normalizza l'immagine Tensor."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, pts=None, den=None):
        if not isinstance(img, torch.Tensor):
             raise TypeError("Normalize si aspetta un Tensor come input per l'immagine")
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, pts, den

class RandomHorizontalFlip(object):
    """Applica flip orizzontale casuale a PIL Image, points, e density."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pts=None, den=None):
        if random.random() < self.p:
            img = F.hflip(img)
            img_w, _ = img.size
            if pts is not None and len(pts) > 0:
                pts[:, 0] = img_w - pts[:, 0]
            if den is not None:
                den = np.fliplr(den).copy()
        return img, pts, den

class RandomResizedCrop(object):
    """Crop casuale e ridimensionamento per PIL Image, points, density."""
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size) if isinstance(size, int) else size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = F.InterpolationMode.BILINEAR

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            aspect_ratio = np.exp(random.uniform(*log_ratio))

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w
        # Fallback
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, pts=None, den=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

        new_pts = None
        if pts is not None and len(pts) > 0:
            mask = (pts[:, 0] >= j) & (pts[:, 0] < j + w) & (pts[:, 1] >= i) & (pts[:, 1] < i + h)
            new_pts = pts[mask].copy()
            if len(new_pts) > 0:
                new_pts[:, 0] = (new_pts[:, 0] - j) * (self.size[1] / w)
                new_pts[:, 1] = (new_pts[:, 1] - i) * (self.size[0] / h)
                new_pts[:, 0] = np.clip(new_pts[:, 0], 0, self.size[1] - 1)
                new_pts[:, 1] = np.clip(new_pts[:, 1], 0, self.size[0] - 1)

        new_den = None
        if den is not None:
            den_cropped = den[i:i+h, j:j+w]
            new_den = cv2.resize(den_cropped, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)

            original_sum = den_cropped.sum()
            resized_sum = new_den.sum()
            if resized_sum > 1e-6:
                 new_den = new_den * (original_sum / resized_sum)
            else:
                 new_den = np.zeros(self.size, dtype=np.float32)

        return img, new_pts, new_den

# --- MODIFICA: SPOSTATA LA DEFINIZIONE QUI ---
class ImageOnlyTransform(object):
    """Wrapper per trasformazioni torchvision che operano solo sull'immagine."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, pts=None, den=None):
        # Applica solo all'immagine (che deve essere PIL o Tensor a seconda della transform)
        img = self.transform(img)
        return img, pts, den
# --- FINE MODIFICA ---

# === FUNZIONE PER COSTRUIRE LA PIPELINE ===

def build_transforms(cfg_data, is_train=True):
    """Costruisce la pipeline di trasformazioni con l'ordine corretto."""
    mean = cfg_data['NORM_MEAN']
    std = cfg_data['NORM_STD']

    if is_train:
        crop_size = cfg_data.get('CROP_SIZE', 256)
        crop_scale_cfg = cfg_data.get('CROP_SCALE', (0.3, 1.0))
        try:
            crop_scale = (float(crop_scale_cfg[0]), float(crop_scale_cfg[1]))
        except (TypeError, ValueError, IndexError):
            crop_scale = (0.3, 1.0)

        # Ordine corretto delle trasformazioni
        return Compose([
            # 1. Geometriche (Input: PIL img, numpy pts, numpy den)
            RandomResizedCrop(size=crop_size, scale=crop_scale),
            RandomHorizontalFlip(p=0.5),

            # 2. Colore/Qualità (Input: PIL img - usiamo il wrapper)
            ImageOnlyTransform(transforms.TrivialAugmentWide()),
            # ImageOnlyTransform(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)), # Alternativa
            # ImageOnlyTransform(transforms.GaussianBlur(kernel_size=3)), # Opzionale

            # 3. Conversione a Tensor (Input: PIL img, Output: Tensor img; Input: numpy den, Output: Tensor den)
            ToTensor(),

            # 4. Normalizzazione (Input: Tensor img)
            Normalize(mean=mean, std=std),
        ])
    else:
        # Validazione/Test: solo ToTensor e Normalize
        return Compose([
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
# P2R_ZIP/datasets/transforms.py
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2
from PIL import Image, ImageFilter

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

class RandomScaleJitter(object):
    """Jitter casuale di scala per robustezza multi-scala."""
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, img, pts=None, den=None):
        scale = random.uniform(*self.scale_range)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w < 10 or new_h < 10:  # Evita immagini troppo piccole
            return img, pts, den
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        if pts is not None and len(pts) > 0:
            pts = pts * scale
        
        if den is not None:
            den = cv2.resize(den, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Conserva l'integrale della densità
            den = den * (scale ** 2)
        
        return img, pts, den

class RandomGaussianNoise(object):
    """Rumore gaussiano per robustezza."""
    def __init__(self, p=0.2, std_range=(0.01, 0.05)):
        self.p = p
        self.std_range = std_range
    
    def __call__(self, img, pts=None, den=None):
        if random.random() < self.p and isinstance(img, Image.Image):
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, random.uniform(*self.std_range), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            img = Image.fromarray((img_array * 255).astype(np.uint8))
        return img, pts, den

class RandomGaussianBlur(object):
    """Gaussian blur casuale."""
    def __init__(self, p=0.2, radius_range=(0.1, 1.5)):
        self.p = p
        self.radius_range = radius_range
    
    def __call__(self, img, pts=None, den=None):
        if random.random() < self.p and isinstance(img, Image.Image):
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img, pts, den

class ImageOnlyTransform(object):
    """Wrapper per trasformazioni torchvision che operano solo sull'immagine."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, pts=None, den=None):
        img = self.transform(img)
        return img, pts, den

def build_transforms(cfg_data, is_train=True):
    """Costruisce la pipeline di trasformazioni con l'ordine corretto."""
    mean = cfg_data['NORM_MEAN']
    std = cfg_data['NORM_STD']

    if is_train:
        crop_size = cfg_data.get('CROP_SIZE', 256)
        crop_scale_cfg = cfg_data.get('CROP_SCALE', (0.5, 1.0))
        try:
            crop_scale = (float(crop_scale_cfg[0]), float(crop_scale_cfg[1]))
        except (TypeError, ValueError, IndexError):
            crop_scale = (0.5, 1.0)

        return Compose([
            # Augmentation geometriche
            RandomScaleJitter(scale_range=(0.9, 1.1)),
            RandomResizedCrop(size=crop_size, scale=crop_scale),
            RandomHorizontalFlip(p=0.5),
            
            # Augmentation visive (solo immagine)
            RandomGaussianNoise(p=0.2, std_range=(0.01, 0.03)),
            ImageOnlyTransform(transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)),
            ImageOnlyTransform(transforms.TrivialAugmentWide()),

            # Conversione e normalizzazione
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    else:
        return Compose([
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
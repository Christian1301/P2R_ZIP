#!/usr/bin/env python3
"""Utility per ricostruire uno split train/val coerente per ShanghaiTech Part A.

Il tool esegue due passi:
1. Riporta (move) eventuali file presenti in ``val_data`` dentro ``train_data``;
2. Seleziona una frazione casuale e sposta immagini + annotazioni corrispondenti
   da ``train_data`` a ``val_data`` mantenendo tutte le possibili estensioni
   (MAT originali, ZIP/P2R NPY).

Esempio d'uso:
    python tools/rebuild_val_split.py \
        --root /mnt/localstorage/cromano/Datasets/ShanghaiTech/part_A \
        --fraction 0.1 --seed 2024

Usare ``--dry-run`` per verificare le operazioni senza modificare i file.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

GT_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("ground_truth", "GT_{name}.mat"),
    ("ground-truth", "GT_{name}.mat"),
    ("labels", "{name}.npy"),
    ("new-anno", "GT_{name}.npy"),
)

def ensure_parent(path: Path) -> None:
    """Crea la cartella padre se non esiste."""
    path.parent.mkdir(parents=True, exist_ok=True)

def move_file(src: Path, dst: Path, dry_run: bool) -> None:
    """Sposta un file gestendo overwrite e modalità dry-run."""
    if not src.exists():
        return
    ensure_parent(dst)
    if dry_run:
        print(f"[DRY-RUN] mv {src} -> {dst}")
        return
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))

def iter_items(directory: Path) -> Iterable[Path]:
    """Restituisce i file nella directory se esiste."""
    if directory.exists():
        yield from sorted(directory.iterdir())

def merge_val_into_train(root: Path, dry_run: bool) -> None:
    """Riporta tutti i file da val_data a train_data (utile per ricominciare)."""
    val_root = root / "val_data"
    if not val_root.exists():
        return
    train_root = root / "train_data"
    print("[INFO] Merge di val_data -> train_data")
    for subdir in ("images",) + tuple(name for name, _ in GT_CANDIDATES):
        src_dir = val_root / subdir
        dst_dir = train_root / subdir
        for item in iter_items(src_dir):
            move_file(item, dst_dir / item.name, dry_run)
        if not dry_run and src_dir.exists():
            try:
                src_dir.rmdir()
            except OSError:
                pass
    if not dry_run:
        try:
            val_root.rmdir()
        except OSError:
            pass

def collect_images(img_dir: Path) -> List[Path]:
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"Nessuna immagine trovata in {img_dir}")
    return images

def gather_ground_truths(base_dir: Path, name: str) -> List[Tuple[Path, Path]]:
    """Trova tutte le annotazioni disponibili per una data immagine."""
    pairs: List[Tuple[Path, Path]] = []
    for subdir, pattern in GT_CANDIDATES:
        src = base_dir / subdir / pattern.format(name=name)
        if src.exists():
            pairs.append((src, Path(subdir) / src.name))
    return pairs

def move_sample(img_path: Path, train_root: Path, val_root: Path, dry_run: bool) -> None:
    """Sposta immagine e annotazioni corrispondenti da train a val."""
    name = img_path.stem
    gts = gather_ground_truths(train_root, name)
    if not gts:
        raise FileNotFoundError(f"Annotazioni mancanti per {img_path}")
    move_file(img_path, val_root / "images" / img_path.name, dry_run)
    for src, rel in gts:
        move_file(src, val_root / rel, dry_run)


def split_train_val(root: Path, fraction: float, seed: int, dry_run: bool) -> None:
    train_root = root / "train_data"
    val_root = root / "val_data"
    val_root.mkdir(parents=True, exist_ok=True)
    (val_root / "images").mkdir(parents=True, exist_ok=True)
    images = collect_images(train_root / "images")
    num_val = max(1, int(len(images) * fraction))
    rng = random.Random(seed)
    rng.shuffle(images)
    val_images = images[:num_val]
    print(f"[INFO] Sposto {len(val_images)} su {len(images)} immagini in val_data")
    for img in val_images:
        move_sample(img, train_root, val_root, dry_run)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ricrea split train/val per ShanghaiTech A")
    parser.add_argument("--root", required=True, type=Path, help="Root del dataset (es. part_A)")
    parser.add_argument("--fraction", type=float, default=0.1, help="Frazione da spostare in val_data")
    parser.add_argument("--seed", type=int, default=2024, help="Seed per la selezione casuale")
    parser.add_argument("--dry-run", action="store_true", help="Mostra operazioni senza eseguirle")
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Non riportare val_data in train_data prima di risplittare",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root: Path = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root dataset non trovata: {root}")
    if not args.skip_merge:
        merge_val_into_train(root, args.dry_run)
    split_train_val(root, args.fraction, args.seed, args.dry_run)
    print("[DONE] Split ricostruito" + (" (dry-run)" if args.dry_run else ""))

if __name__ == "__main__":
    main()

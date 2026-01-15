#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2RLoss Originale dal paper "Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting"

Questa è la loss originale del paper P2R, adattata per funzionare con l'architettura P2R-ZIP.

La loss funziona così:
1. Per ogni punto GT, trova la cella più vicina nella density map
2. Assegna un target binario (1 = persona presente)
3. Calcola BCE con weight dinamici basati sul matching

Differenze chiave dalla semplice count loss:
- Vincolo SPAZIALE: il modello deve predire DOVE sono le persone
- Matching OTTIMALE: ogni punto GT viene assegnato alla cella migliore
- Weight DINAMICI: bilancia automaticamente regioni dense vs sparse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-10


class L2Distance:
    """Calcola distanza L2 tra coordinate predette e GT."""
    
    def __init__(self, factor=512):
        self.factor = factor

    def __call__(self, X, Y):
        """
        Args:
            X: [1, N, 2] coordinate delle celle
            Y: [1, M, 2] coordinate dei punti GT
        Returns:
            C: [1, N, M] matrice delle distanze normalizzate
        """
        x_col = X.unsqueeze(-2)  # [1, N, 1, 2]
        y_row = Y.unsqueeze(-3)  # [1, 1, M, 2]
        C = torch.norm(x_col - y_row, dim=-1)  # [1, N, M]
        C = C / self.factor
        return C


class P2RLossOriginal(nn.Module):
    """
    Point-to-Region Loss originale dal paper.
    
    Questa loss è fondamentale per evitare overfitting perché:
    1. Non basta predire il count corretto - serve predire la POSIZIONE
    2. Il matching point-to-region crea vincoli spaziali forti
    3. I weight dinamici impediscono al modello di ignorare regioni difficili
    
    Args:
        factor: fattore di normalizzazione per le distanze (default: 1)
        min_radius: raggio minimo per il matching (default: 8)
        max_radius: raggio massimo per il matching (default: 96)
        cost_class: peso del costo di classificazione (default: 1)
        cost_point: peso del costo spaziale (default: 8)
    """
    
    def __init__(
        self, 
        factor=1, 
        min_radius=8, 
        max_radius=96,
        cost_class=1,
        cost_point=8,
        reduction='mean'
    ):
        super().__init__()
        self.factor = factor
        self.cost = L2Distance(1)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.reduction = reduction

    def forward(self, density, points_list, down_rate, masks=None, crop_den_masks=None):
        """
        Calcola la P2R loss.
        
        Args:
            density: [B, 1, H, W] or [B, H, W] density map predictions
            points_list: lista di tensori [N, 2+] con coordinate GT (y, x, ...)
            down_rate: fattore di downsampling (es. 8)
            masks: maschere opzionali per semi-supervised (default: None)
            crop_den_masks: maschere di crop opzionali (default: None)
            
        Returns:
            loss: scalar tensor
        """
        # Assicura che density sia [B, H, W, 1] per compatibilità con codice originale
        if density.dim() == 4:
            density = density.squeeze(1)  # [B, H, W]
        
        bs = len(points_list)
        total_loss = 0.0
        
        for i in range(bs):
            den = density[i]  # [H, W]
            seq = points_list[i]  # [N, 2+] - formato (y, x, ...) o (x, y, ...)
            
            if crop_den_masks is not None:
                crop_den_mask = crop_den_masks[i]
            else:
                crop_den_mask = None
                
            if masks is not None:
                mask = masks[i]
            else:
                mask = None
            
            # Reshape density per il calcolo
            den = den.unsqueeze(-1)  # [H, W, 1]
            H, W = den.shape[:2]
            
            if seq.size(0) < 1:
                # Nessun punto GT: target tutto zero con weight ridotto
                target = torch.zeros_like(den)
                weight = torch.ones_like(den) * 0.5
                loss_i = F.binary_cross_entropy_with_logits(den, target, weight=weight)
            else:
                # Costruisci griglia di coordinate delle celle
                # A_coord: [1, H*W, 2] - coordinate (y, x) del centro di ogni cella
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=den.device),
                    torch.arange(W, device=den.device),
                    indexing='ij'
                )
                A_coord = torch.stack([grid_y, grid_x], dim=-1).view(1, -1, 2).float()
                A_coord = A_coord * down_rate + (down_rate - 1) / 2  # Centro della cella
                
                # A: [1, H*W, 1] - valori della density map
                A = den.view(1, -1, 1)
                
                # B_coord: [1, M, 2] - coordinate dei punti GT
                # Il formato dipende da come sono salvati i punti
                # Assumiamo (y, x) come nel codice originale
                if seq.shape[1] >= 2:
                    B_coord = seq[None, :, :2].float()  # [1, M, 2]
                else:
                    B_coord = seq[None, :, :].float()
                
                # B: [1, M, 1] - tutti 1 (ogni punto è una persona)
                B = torch.ones(seq.size(0), device=den.device).float().view(1, -1, 1)
                
                # Maschera opzionale per semi-supervised
                if mask is not None:
                    MB = mask.view_as(B).to(B)
                
                with torch.no_grad():
                    # Calcola matrice delle distanze
                    C = self.cost(A_coord, B_coord)  # [1, H*W, M]
                    
                    # Per ogni cella, trova il punto GT più vicino
                    minC, mcidx = C.min(dim=-1, keepdim=True)  # [1, H*W, 1]
                    
                    # M: matrice di matching - 1 se la cella è assegnata a quel punto
                    M = torch.zeros_like(C).scatter_(-1, mcidx, 1.0)
                    M = M * (C < self.max_radius)  # Solo se entro max_radius
                    
                    # Calcola il raggio adattivo per ogni cella
                    maxC = (minC.view_as(A) * M).amax(dim=1, keepdim=True)
                    maxC = torch.clip(maxC, min=self.min_radius, max=self.max_radius)
                    C_normalized = C / maxC
                    
                    # Costo combinato: spaziale + classificazione
                    C_cost = C_normalized * self.cost_point - A * self.cost_class
                    
                    # Filtra punti validi (che hanno almeno una cella assegnata)
                    valid_points = (M.sum(dim=1) > 0).view(-1)
                    C_cost = C_cost[..., valid_points]
                    M = M[..., valid_points]
                    B = B[:, valid_points, :]
                    B_coord = B_coord[:, valid_points, :]
                    
                    # Trova il matching ottimale per ogni punto
                    C2 = M * C_cost + (1 - M) * (C_cost.max() + 1)
                    minC2, mcidx2 = C2.min(dim=1, keepdim=True)
                    
                    # T: target binario - 1 se la cella è il match ottimale per qualche punto
                    T = torch.zeros_like(C2).scatter_(1, mcidx2, 1.0).sum(dim=-1).view(1, -1, 1)
                    T = (T > 0.5).to(A).view_as(A)
                    
                    # W: weight per BCE - celle con target=1 pesano di più
                    W = T + 1  # weight 2 per positivi, 1 per negativi
                    
                    # Applica maschera semi-supervised se presente
                    if mask is not None:
                        M_weight = (M @ MB[:, valid_points, :]) + 1 - M.sum(dim=-1).view_as(A)
                        W = W * M_weight
                
                # Applica maschera di crop se presente
                if crop_den_mask is not None:
                    W = W * crop_den_mask.view_as(W)
                
                # Calcola BCE loss
                loss_i = F.binary_cross_entropy_with_logits(A, T, weight=W)
            
            total_loss = total_loss + loss_i
        
        # Media sul batch
        loss = total_loss / bs
        return loss


class P2RLossForZIP(nn.Module):
    """
    Wrapper della P2RLoss che gestisce l'output del P2R-ZIP model.
    
    Il modello P2R-ZIP produce density map continue, mentre P2RLoss originale
    lavora con logits. Questo wrapper:
    1. Converte le coordinate dal formato (x, y) a (y, x) se necessario
    2. Gestisce il downsampling corretto
    3. Restituisce anche metriche utili (MAE, count predictions)
    """
    
    def __init__(
        self,
        min_radius=8,
        max_radius=96,
        cost_class=1,
        cost_point=8
    ):
        super().__init__()
        self.p2r_loss = P2RLossOriginal(
            factor=1,
            min_radius=min_radius,
            max_radius=max_radius,
            cost_class=cost_class,
            cost_point=cost_point
        )
    
    def forward(self, density, points_list, down_rate, return_metrics=False):
        """
        Args:
            density: [B, 1, H, W] density predictions
            points_list: lista di tensori con coordinate GT
            down_rate: fattore di downsampling
            return_metrics: se True, restituisce anche MAE e predictions
            
        Returns:
            loss: scalar
            (opzionale) metrics: dict con 'mae', 'gt_counts', 'pred_counts'
        """
        # Converti density in logits (inverse sigmoid approssimato)
        # Clamp per evitare log(0) o log(1)
        density_clamped = torch.clamp(density, min=1e-6, max=1-1e-6)
        # logits = torch.log(density_clamped / (1 - density_clamped))
        
        # In realtà, per P2R il modello dovrebbe outputtare logits direttamente
        # Usiamo la density così com'è e la loss farà sigmoid internamente
        
        # Per P2RLoss, passiamo la density come logits
        # Il modello P2R originale outputta logits, non probabilità
        loss = self.p2r_loss(density, points_list, down_rate)
        
        if return_metrics:
            with torch.no_grad():
                # Conta pixel > 0 come nel paper originale
                pred_counts = []
                gt_counts = []
                
                for i, pts in enumerate(points_list):
                    # Predizione: conta celle con density > 0 (dopo sigmoid)
                    pred = (torch.sigmoid(density[i]) > 0.5).sum().item()
                    # In alternativa, conta celle positive
                    # pred = (density[i] > 0).sum().item()
                    pred_counts.append(pred)
                    gt_counts.append(len(pts))
                
                mae = sum(abs(p - g) for p, g in zip(pred_counts, gt_counts)) / len(gt_counts)
                
                metrics = {
                    'mae': mae,
                    'gt_counts': gt_counts,
                    'pred_counts': pred_counts
                }
                return loss, metrics
        
        return loss


# Alias per retrocompatibilità
P2RLoss = P2RLossOriginal
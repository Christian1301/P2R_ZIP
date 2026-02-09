# P2R_ZIP/losses/zip_nll.py
import torch
import torch.nn.functional as F

def zip_nll(pi, lam, target_counts, eps=1e-8, reduction="mean"):
    """
    Zero-Inflated Poisson Negative Log-Likelihood per blocco.
    Inputs:
      pi:   prob. blocco OCCUPATO in [0,1], shape [B,1,Hb,Wb]
      lam:  rate Poisson >=0, shape [B,1,Hb,Wb]
      target_counts: int >=0 per blocco, shape [B,1,Hb,Wb]
    Formula:
      se y==0:   log( (1-pi) + pi * e^{-lam} )
      se y>0:    log( pi * e^{-lam} * lam^y / y! )
    """
    
    target_h, target_w = target_counts.shape[-2:]
    if pi.shape[-2:] != (target_h, target_w):
        pi = F.interpolate(pi, size=(target_h, target_w), mode='bilinear', align_corners=False)
    if lam.shape[-2:] != (target_h, target_w):
        lam = F.interpolate(lam, size=(target_h, target_w), mode='bilinear', align_corners=False)

    pi = torch.clamp(pi, 0.0 + eps, 1.0 - eps)
    lam = torch.clamp(lam, eps, 1e6)
    y = target_counts

    is_zero = (y == 0).float()
    is_pos  = 1.0 - is_zero

    # log p(y=0)
    log_p0 = torch.log((1.0 - pi) + pi * torch.exp(-lam) + eps)

    # log p(y>0) = log(pi) - lam + y*log(lam) - log(y!)
    log_pi = torch.log(pi + eps)
    log_fact = torch.lgamma(y + 1.0)  # log(y!)
    log_py = log_pi - lam + y * torch.log(lam + eps) - log_fact

    # Ora le dimensioni sono garantite essere uguali
    nll = - (is_zero * log_p0 + is_pos * log_py)
    
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll
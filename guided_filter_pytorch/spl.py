import torch

from .boxfilter import boxfilter2d

def spl_gray(guide, src, radius, eps, closed_form=False):
    """
    Structure Preservation loss (SPL) for grayscale images.
    
    Args:
        guide (torch.Tensor): Guidance image tensor of shape (B, 1, H, W).
        src (torch.Tensor): Source image tensor of shape (B, 1, H, W).
        radius (int): Radius of the filter.
        eps (float): Regularization term.
        closed_form (bool): If True, use closed-form solution.
        
    Returns:
        If closed_form is True, returns the closed-form solution that minimizes the structure preservation loss.
        If closed_form is False, returns loss value
    """
    
    if guide.ndim == 3:
        guide = guide[:, None]
    if src.ndim == 3:
        src = src[:, None]
    # if guide.shape[1] != 1 or src.shape[1] != 1:
    #     raise ValueError("Both guide and src must have 1 channel (grayscale).")
    
    ones = torch.ones_like(guide)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N
    mean_p = boxfilter2d(src, radius) / N
    mean_Ip = boxfilter2d(guide*src, radius) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = boxfilter2d(guide*guide, radius) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    a = a.clone().detach()
    b = b.clone().detach()

    mean_pp = boxfilter2d(src * src, radius) / N
    aa = a*a
    bb = b*b

    inv_w = 1.0

    var_p = mean_pp - mean_p * mean_p
    a_inv = cov_Ip / (var_p + eps)
    b_inv = mean_I - a_inv * mean_p

    a_inv = a_inv.clone().detach()
    b_inv = b_inv.clone().detach()

    aa_inv = a_inv * a_inv
    bb_inv = b_inv * b_inv

    # L2 version
    # Equation (5) in the paper
    # L_SPL = mean(mean((a*I + b - p)^2) + mean((a_inv*p + b_inv - I)^2))
    dir_diff = mean_pp + aa*mean_II + bb - 2*a*mean_Ip - 2*b*mean_p + 2*a*b*mean_I
    dir_diff_inv = mean_II + aa_inv*mean_pp + bb_inv - 2*a_inv*mean_Ip - 2*b_inv*mean_I + 2*a_inv*b_inv*mean_p
    spl = dir_diff + inv_w * dir_diff_inv
    mean_spl = torch.mean(spl)

    if closed_form:
        denom = boxfilter2d(aa + ones, radius) / N
        A = boxfilter2d(a + a_inv, radius) / N
        B = boxfilter2d(b_inv - a * b, radius) / N
        solution = (A * src + B) / (denom + 1e-6)
        return mean_spl, solution
    else:
        return mean_spl


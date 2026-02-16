import torch
import torch.nn
import kornia
from typing import Optional, Union
from guided_filter_pytorch.spl import spl_gray

class Loss:
    def __init__(self, opt_start_step, opt_iter, post_opt, weight=1.0, mask: Optional[torch.Tensor] = None):
        self.opt_start_step = opt_start_step
        self.opt_iter = opt_iter
        self.post_opt = post_opt
        self.weight = weight
        self.mask = mask

    # Common preprocessing: normalization & mask application
    def preprocess(self, source, target):
        source = (source.to(dtype=torch.float32) + 1.) / 2.
        target = (target.to(dtype=torch.float32) + 1.) / 2.
        if self.mask is not None:
            assert source.shape[2:] == self.mask.shape, f'source shape: {source.shape}, mask shape: {self.mask.shape}'
            source = self.mask * source
            target = self.mask * target
        return source, target

    def __call__(self, source, target):
        s, t = self.preprocess(source, target)
        return self.weight * self.compute_loss(s, t)

    def compute_loss(self, s, t):
        raise NotImplementedError("compute_loss must be implemented in subclasses")


class L1(Loss):
    def __init__(self, opt_start_step, opt_iter, post_opt, weight=1000.0, mask: Optional[torch.Tensor] = None):
        super().__init__(opt_start_step, opt_iter, post_opt, weight, mask)
        self.l = torch.nn.L1Loss()

    def compute_loss(self, s, t):
        return self.l(s, t)


class MSE(Loss):
    def __init__(self, opt_start_step, opt_iter, post_opt, weight=1000.0, mask: Optional[torch.Tensor] = None):
        super().__init__(opt_start_step, opt_iter, post_opt, weight, mask)
        self.l = torch.nn.MSELoss()

    def compute_loss(self, s, t):
        return self.l(s, t)


class SPL(Loss):
    def __init__(self, opt_start_step, opt_iter, post_opt, weight=10000.0, radius=5, eps=1e-4, mask: Optional[torch.Tensor] = None):
        super().__init__(opt_start_step, opt_iter, post_opt, weight, mask)
        self.radius = radius
        self.eps = eps

    def compute_loss(self, s, t):
        s = torch.mean(s, 1, True)
        t = torch.mean(t, 1, True)
        return spl_gray(s, t, self.radius, self.eps, closed_form=False)


class CPL(Loss):
    # Note: The paper states lambda=1e-4, but the correct value is ~1e-1 relative to SPL weight.
    def __init__(self, opt_start_step, opt_iter, post_opt, weight=1000.0, mask: Optional[torch.Tensor] = None):
        super().__init__(opt_start_step, opt_iter, post_opt, weight, mask)
        self.l = torch.nn.MSELoss()

    def compute_loss(self, s, t):
        s = kornia.color.rgb_to_ycbcr(s)[:, 1:]
        t = kornia.color.rgb_to_ycbcr(t)[:, 1:]
        return self.l(s, t)


class CompositeLoss(Loss):
    def __init__(self, opt_start_step, opt_iter, post_opt, structure_loss, color_loss):
        super().__init__(opt_start_step, opt_iter, post_opt)
        self.structure_loss = structure_loss
        self.color_loss = color_loss

    def __call__(self, source, target):
        return self.structure_loss(source, target) + self.color_loss(source, target)


def is_none(val):
    return val is None or (isinstance(val, str) and val.strip().lower() in ["", "none"])


def _get_structure_loss(name, opt_start_step, opt_iter, post_opt, structure_weight, structure_mask):
    mapping = {
        'L1': lambda: L1(opt_start_step, opt_iter, post_opt, weight=structure_weight, mask=structure_mask),
        'MSE': lambda: MSE(opt_start_step, opt_iter, post_opt, weight=structure_weight, mask=structure_mask),
        'SPL': lambda: SPL(opt_start_step, opt_iter, post_opt, weight=structure_weight, mask=structure_mask),
    }
    if name not in mapping:
        raise ValueError(f"Undefined structure loss: {name}")
    return mapping[name]()


def _get_color_loss(name, opt_start_step, opt_iter, post_opt, color_weight, color_mask):
    mapping = {
        'MSE': lambda: MSE(opt_start_step, opt_iter, post_opt, weight=color_weight, mask=color_mask),
        'CPL': lambda: CPL(opt_start_step, opt_iter, post_opt, weight=color_weight, mask=color_mask),
    }
    if name not in mapping:
        raise ValueError(f"Undefined color loss: {name}")
    return mapping[name]()


def init_loss(structure_loss_name: str,
              color_loss_name: str,
              opt_start_step: float,
              opt_iter: int,
              post_opt: bool,
              structure_weight: float = 10000.0,
              color_weight: float = 10000.0,
              structure_mask: Optional[torch.Tensor] = None,
              color_mask: Optional[torch.Tensor] = None) -> Union[Loss, CompositeLoss, None]:
    """
    Initializes loss functions for structure and color domains.

    Parameters:
      structure_loss_name (str): Loss type for structure (e.g. "SPL")
      color_loss_name (str): Loss type for color (e.g. "CPL")
      opt_start_step (float): common optimization start step
      opt_iter (int): common optimization iterations
      post_opt (bool): common post-optimization flag
      structure_weight (float): weight for structure loss
      structure_mask (Optional[Tensor]): mask for structure loss
      color_weight (float): weight for color loss
      color_mask (Optional[Tensor]): mask for color loss

    Returns:
      CompositeLoss instance if both losses are provided,
      or a single Loss instance if only one is provided,
      or None if neither is provided.
    """
    s_loss = None if is_none(structure_loss_name) else _get_structure_loss(structure_loss_name, opt_start_step, opt_iter, post_opt,
                                                                            structure_weight, structure_mask)
    c_loss = None if is_none(color_loss_name) else _get_color_loss(color_loss_name, opt_start_step, opt_iter, post_opt,
                                                                    color_weight, color_mask)

    if s_loss is None and c_loss is None:
        return None
    if s_loss is None:
        return c_loss
    if c_loss is None:
        return s_loss
    return CompositeLoss(opt_start_step, opt_iter, post_opt, s_loss, c_loss)

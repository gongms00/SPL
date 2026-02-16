import torch
from guided_filter_pytorch.guided_filter import GuidedFilter2d
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as nnf
import ptp_utils

def pil_to_torch(image):
    if image.mode == "RGB":
        RGB_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()[None] / 255.
        I_image = torch.mean(RGB_image, dim=1, keepdim=True)
        return I_image
    elif image.mode == "L":
        return torch.from_numpy(np.array(image)).float()[None, None] / 255.
    elif image.mode == "I": # Assume 16-bit depth
        return torch.from_numpy(np.array(image)).float()[None, None] / 65535.

def torch_to_pil(tensor):
    if tensor.shape[1] == 1:
        return Image.fromarray((tensor[0, 0].cpu().numpy() * 255).astype(np.uint8))
    else:
        return Image.fromarray((tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def guided_upsample(mask_tensor, guide_tensor, device):
    upsample_multiplier = 2
    radius = 1
    radius_add = 1
    eps = 1e-2

    mask_image = torch_to_pil(mask_tensor)
    guide_image = torch_to_pil(guide_tensor)

    cur_size = mask_tensor.shape[-1] * upsample_multiplier
    end_size = guide_tensor.shape[-1]

    # repeat upsampling source tensor by 2 until it is same or larger than guide tensor
    while cur_size <= end_size:
        mask_image_resized = mask_image.resize((cur_size, cur_size), Image.LANCZOS)
        guide_image_resized = guide_image.resize((cur_size, cur_size), Image.LANCZOS)

        mask_tensor_resized = pil_to_torch(mask_image_resized).to(device=device)
        guide_tensor_resized = pil_to_torch(guide_image_resized).to(device=device)

        GF = GuidedFilter2d(radius=radius, eps=eps, color_seperate=False)
        output_tensor = GF(mask_tensor_resized, guide_tensor_resized)
        output_image = torch_to_pil(output_tensor)

        mask_image = output_image
        cur_size *= upsample_multiplier
        radius += radius_add

    return output_tensor

def gamma_correction_below_threshold(tensor, threshold, gamma):

    corrected_tensor = tensor.clone()
    mask = corrected_tensor < threshold
    corrected_tensor[mask] = torch.pow(corrected_tensor[mask] / float(threshold), gamma) * threshold
    return corrected_tensor

def refine_mask(mask_tensor):
    # Assume shape of mask tensor is (H, W) and values are in [0, 1]
    mask_np = (mask_tensor * 255).cpu().numpy().astype(np.uint8)

    # Step 1: Apply adaptive thresholding and consider black (0) areas as the edges
    edges_np = cv2.adaptiveThreshold(
        mask_np, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY, 
        blockSize=15,
        C=5
    )

    # Step 2: Calculate mean of the edges and set gamma threshold
    edges_np = edges_np == 0
    if edges_np.any():
        mean_edges = int(mask_np[edges_np].mean())
    else:
        mean_edges = int(mask_np.mean())
    
    gamma_threshold = (mean_edges + 20) / 255.

    # Step 3: Apply gamma correction to the mask tensor
    refined_mask_tensor = gamma_correction_below_threshold(mask_tensor, gamma_threshold, 4.0)
    return refined_mask_tensor

def parse_mask_words(text):
    """Parse comma-separated mask words string into a list, or None if empty."""
    if isinstance(text, str):
        words = [w for w in text.replace(' ', '').split(',') if w]
        return words if words else None
    return None


def _get_rough_mask(prompts, tokenizer, controller, select, thresh, mask_words, torch_dtype, device):
    """Extract a rough 16x16 attention mask from the controller."""
    if mask_words is not None:
        return ptp_utils.get_rough_mask(
            prompts, tokenizer, controller, 16, ["up", "down"], select, thresh, mask_words)
    return torch.zeros(16, 16, dtype=torch_dtype, device=device)


def build_mask(source_image, target_image, controller, prompts, tokenizer,
               mask_words_src, mask_thresh_src, invert_mask_src,
               mask_words_tgt, mask_thresh_tgt, invert_mask_tgt,
               torch_dtype, device):
    """Build upsampled mask from cross-attention maps.

    Args:
        source_image: Source PIL image (for source mask upsampling guide).
        target_image: Target PIL image from initial pipe run (for target mask upsampling guide).
        controller: AttentionRefine controller with stored attention maps.
        prompts: [source_prompt, target_prompt] list.
        tokenizer: Tokenizer for word index extraction.
        mask_words_src/tgt: Comma-separated mask word strings or None.
        mask_thresh_src/tgt: Threshold values for rough mask extraction.
        invert_mask_src/tgt: Whether to invert the respective masks.
        torch_dtype: Torch dtype for tensors.
        device: Torch device.

    Returns:
        Combined upsampled mask tensor (H, W).
    """
    mask_words_src = parse_mask_words(mask_words_src)
    mask_words_tgt = parse_mask_words(mask_words_tgt)

    rough_mask_src = _get_rough_mask(prompts, tokenizer, controller, 0, mask_thresh_src, mask_words_src, torch_dtype, device)
    rough_mask_tgt = _get_rough_mask(prompts, tokenizer, controller, 1, mask_thresh_tgt, mask_words_tgt, torch_dtype, device)

    upsample_mask_src = get_upsampled_mask(
        source_image, "guided", rough_mask_src, mask_words_src, device, not invert_mask_src)
    upsample_mask_tgt = get_upsampled_mask(
        target_image, "guided", rough_mask_tgt, mask_words_tgt, device, not invert_mask_tgt)

    return torch.maximum(upsample_mask_src, upsample_mask_tgt)


def get_upsampled_mask(target_image, upsample_method, rough_mask_, mask_words_, device, preserve_selected=True):
    upsample_mask_ = torch.zeros(target_image.height, target_image.width, dtype=rough_mask_.dtype, device=device)
    if mask_words_ is not None:
        if upsample_method == "bilinear":
            upsample_mask_ = nnf.interpolate(rough_mask_.unsqueeze(0).unsqueeze(0), size=(target_image.height, target_image.width), mode='bilinear', align_corners=False).squeeze()
        elif upsample_method == "bicubic":
            upsample_mask_ = nnf.interpolate(rough_mask_.unsqueeze(0).unsqueeze(0), size=(target_image.height, target_image.width), mode='bicubic', align_corners=False).squeeze()
        elif upsample_method == "guided":
            img_tensor = pil_to_torch(target_image).to(device=device)
            upsample_mask_ = guided_upsample(rough_mask_.unsqueeze(0).unsqueeze(0), img_tensor, device=device).squeeze()
        if not preserve_selected:
            upsample_mask_ = 1.0 - upsample_mask_
        
        if upsample_method == "guided":
            upsample_mask_ = refine_mask(upsample_mask_)  # Refine mask using adaptive thresholding and gamma correction
    return upsample_mask_


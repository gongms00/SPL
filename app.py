from diffusers import LCMScheduler
from pipeline_spl import EditPipeline
import gradio as gr
import torch
from PIL import Image
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Dict
import abc
import ptp_utils
import numpy as np
import seq_aligner
import random
import mask_upsample
from losses import init_loss

LOW_RESOURCE = False
MAX_NUM_WORDS = 77

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id_or_path = "SimianLuo/LCM_Dreamshaper_v7"
device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = LCMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
pipe = EditPipeline.from_pretrained(model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype)

tokenizer = pipe.tokenizer
encoder = pipe.text_encoder

if torch.cuda.is_available():
    pipe = pipe.to("cuda")


class LocalBlend:

    def get_mask(self,x_t,maps,word_idx, thresh, i):
        maps = maps * word_idx.reshape(1,1,1,1,-1)
        maps = (maps[:,:,:,:,1:self.len-1]).mean(0,keepdim=True)
        maps = (maps).max(-1)[0]
        maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = maps > thresh
        return mask


    def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False):
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        h,w = x_t.shape[2],x_t.shape[3]
        h , w = ((h+1)//2+1)//2, ((w+1)//2+1)//2
        maps = [item.reshape(2, -1, 1, h // int((h*w/item.shape[-2])**0.5),  w // int((h*w/item.shape[-2])**0.5), MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps_s = maps[0,:]
        maps_m = maps[1,:]
        thresh_e = temperature / alpha_prod ** (0.5)
        if thresh_e < self.thresh_e:
          thresh_e = self.thresh_e
        thresh_m = self.thresh_m
        mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
        mask_m = self.get_mask(x_t, maps_s, (self.alpha_m-self.alpha_me), thresh_m, i)
        mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)
        if self.alpha_e.sum() == 0:
          x_t_out = x_t
        else:
          x_t_out = torch.where(mask_e, x_t, x_m)
        x_t_out = torch.where(mask_m, x_s, x_t_out)
        if use_xm:
          x_t_out = torch.where(mask_me, x_m, x_t_out)

        return x_m, x_t_out

    def __init__(self,thresh_e=0.3, thresh_m=0.3):
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m

    def set_map(self, ms, alpha, alpha_e, alpha_m,len):
        self.m = ms
        self.alpha = alpha
        self.alpha_e = alpha_e
        self.alpha_m = alpha_m
        alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
        self.alpha_me = alpha_me.to(torch.float)
        self.len = len


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn
    def self_attn_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self,i, t, x_s, x_t, x_m, alpha_prod):
        if (self.local_blend is not None) and (i>0):
            use_xm = (self.cur_step+self.start_steps+1 == self.num_steps)
            x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
        return x_m, x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out

    def self_attn_forward(self, q, k, v, num_heads):
        # batch dimension : [source * num_heads , target * num_heads, mutual * num_heads]
        if q.shape[0]//num_heads == 3:
            if (self.self_replace_steps <= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ): # Later step
                q=torch.cat([q[:num_heads*2],q[num_heads:num_heads*2]])     # [source, target, target]
                k=torch.cat([k[:num_heads*2],k[:num_heads]])                # [source, target, source]
                v=torch.cat([v[:num_heads*2],v[:num_heads]])                # [source, target, source]
            else:                                                                                       # Early step
                q=torch.cat([q[:num_heads],q[:num_heads],q[:num_heads]])    # [source, source, source]
                k=torch.cat([k[:num_heads],k[:num_heads],k[:num_heads]])    # [source, source, source]
                v=torch.cat([v[:num_heads*2],v[:num_heads]])                # [source, target, source]
            return q,k,v
        else:
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            if (self.self_replace_steps <= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ):
                qu=torch.cat([qu[:num_heads*2],qu[num_heads:num_heads*2]])
                qc=torch.cat([qc[:num_heads*2],qc[num_heads:num_heads*2]])
                ku=torch.cat([ku[:num_heads*2],ku[:num_heads]])
                kc=torch.cat([kc[:num_heads*2],kc[:num_heads]])
                vu=torch.cat([vu[:num_heads*2],vu[:num_heads]])
                vc=torch.cat([vc[:num_heads*2],vc[:num_heads]])
            else:
                qu=torch.cat([qu[:num_heads],qu[:num_heads],qu[:num_heads]])
                qc=torch.cat([qc[:num_heads],qc[:num_heads],qc[:num_heads]])
                ku=torch.cat([ku[:num_heads],ku[:num_heads],ku[:num_heads]])
                kc=torch.cat([kc[:num_heads],kc[:num_heads],kc[:num_heads]])
                vu=torch.cat([vu[:num_heads*2],vu[:num_heads]])
                vc=torch.cat([vc[:num_heads*2],vc[:num_heads]])

            return torch.cat([qu, qc], dim=0) ,torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross :
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size,h,  *attn.shape[1:])
            attn_base, attn_repalce,attn_masa = attn[0], attn[1], attn[2]   # source, target, mutual
            attn_replace_new = self.replace_cross_attention(attn_masa, attn_repalce)
            attn_base_store = self.replace_cross_attention(attn_base, attn_repalce)

            # Replace target cross attention with source cross attention
            if (self.cross_replace_steps >= ((self.cur_step+self.start_steps+1)*1.0 / self.num_steps) ):
                attn[1] = attn_base_store
            # Store original attention for mask extraction
            attn_store = torch.cat([attn_base, attn_repalce])
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])

            super(AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
        return attn

    def __init__(self, prompts, num_steps: int,start_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)+1
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.num_steps=num_steps
        self.start_steps=start_steps
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device).to(torch_dtype)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_masa, att_replace):
        attn_masa_replace = attn_masa[:, :, self.mapper].squeeze()
        attn_replace = attn_masa_replace * self.alphas + \
                 att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, prompt_specifiers, num_steps: int,start_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps,start_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers, tokenizer, encoder, device)

        self.mapper, alphas, ms = self.mapper.to(device), alphas.to(device).to(torch_dtype), ms.to(device).to(torch_dtype)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
        ms = ms.to(device)
        alpha_e = alpha_e.to(device)
        alpha_m = alpha_m.to(device)
        t_len = len(tokenizer(prompts[1])["input_ids"])
        self.local_blend.set_map(ms,alphas,alpha_e,alpha_m,t_len)


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch_dtype)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def replace_nsfw_images(results):
    if results.nsfw_content_detected is not None:
        for i in range(len(results.images)):
            if results.nsfw_content_detected[i]:
                results.images[i] = Image.open("assets/nsfw.png")
    return results.images[0]


def preprocess_image(img, size=512):
    """Resize and center crop to size x size."""
    ratio = max(size / img.width, size / img.height)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))


def inference(img, source_prompt, target_prompt,
          guidance_s, guidance_t,
          num_inference_steps, seed,
          attn_control_steps, opt_start_step,
          enable_spl=False, spl_area="Whole image",
          enable_cpl=False, cpl_area="Whole image",
          structure_weight=10000, color_weight=10000,
          opt_iter=100, post_opt=False,
          mask_words_src=None, mask_thresh_src=0.5, invert_mask_src=False,
          mask_words_tgt=None, mask_thresh_tgt=0.5, invert_mask_tgt=False,
          random_seed=False):

    if random_seed:
        seed = random.randint(0, 2147483647)

    img = preprocess_image(img)

    num_start = 0
    local_blend = LocalBlend(thresh_e=0.6, thresh_m=0.6)

    def run_pipe(loss_fn=None):
        torch.manual_seed(seed)
        controller = AttentionRefine(
            [source_prompt, target_prompt], [["", ""]], num_inference_steps, num_start,
            cross_replace_steps=attn_control_steps, self_replace_steps=attn_control_steps,
            local_blend=local_blend)
        ptp_utils.register_attention_control(pipe, controller)
        output = pipe(
            prompt=target_prompt, source_prompt=source_prompt,
            positive_prompt="", negative_prompt="",
            image=img, num_inference_steps=num_inference_steps, eta=1,
            strength=1, guidance_scale=guidance_t, source_guidance_scale=guidance_s,
            denoise_model=False, callback=controller.step_callback, loss_fn=loss_fn)
        return controller, output

    # Build upsampled mask if needed
    apply_mask_structure = enable_spl and spl_area == "Masked area"
    apply_mask_color = enable_cpl and cpl_area == "Masked area"
    mask = None

    if apply_mask_structure or apply_mask_color:
        controller, output_temp = run_pipe()
        mask = mask_upsample.build_mask(
            img, output_temp.images[0], controller,
            [source_prompt, target_prompt], tokenizer,
            mask_words_src, mask_thresh_src, invert_mask_src,
            mask_words_tgt, mask_thresh_tgt, invert_mask_tgt,
            torch_dtype, device)

    # Build loss function and run final pipe
    structure_loss = "SPL" if enable_spl else None
    color_loss = "CPL" if enable_cpl else None
    structure_mask = mask if apply_mask_structure else None
    color_mask = mask if apply_mask_color else None
    loss_fn = init_loss(structure_loss, color_loss, opt_start_step, opt_iter, post_opt,
                        structure_weight, color_weight, structure_mask, color_mask)

    _, results = run_pipe(loss_fn=loss_fn)

    # Build gallery output
    output_img = replace_nsfw_images(results)
    gallery = [(output_img, "Output")]
    if mask is not None:
        gallery.append((ptp_utils.tensor_to_pil_image(mask), "Upsampled mask"))
    return gallery



css = """
.gradio-container {max-width: 1200px !important; margin: auto !important}
.tabs {margin-top: 0; margin-bottom: 0}
#gallery {min-height: 20rem}
"""

with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # Edge-Aware Image Manipulation with Structure-Preservation Loss
        <p>
            <a href="https://gongms00.github.io/SPL-project-page/" target="_blank">Project Page</a> |
            <a href="https://arxiv.org/abs/2601.16645" target="_blank">Paper (WACV 2026)</a> |
            <a href="https://github.com/gongms00/SPL" target="_blank">Code</a>
        </p>

        A training-free guidance technique for latent diffusion models that preserves edge structures and colors during text-driven image editing.

        **Quick start:** Upload an image, enter source/target prompts, then click **Run**.
        Enable **Preserve structure (SPL)** or **Preserve color (CPL)** to apply the proposed losses.
        
        **Note:** Input images are resized and center-cropped to 512x512. Only 512x512 resolution is supported.
        """
    )
    with gr.Row():

        with gr.Column(scale=55):
            with gr.Group():
                img = gr.Image(label="Input image", height=512, type="pil")
                gallery_out = gr.Gallery(label="Results", height=512, columns=2, object_fit="contain")

        with gr.Column(scale=45):
            with gr.Tab("Edit"):
                with gr.Group():
                    source_prompt = gr.Textbox(label="Source prompt", value="", placeholder="Source prompt describes the input image")
                    target_prompt = gr.Textbox(label="Target prompt", value="", placeholder="Target prompt describes the output image")
                    attn_control_steps = gr.Slider(label="Attention control schedule", value=0.8, minimum=0.0, maximum=1, step=0.05,
                                                   info="Higher = more consistency with the source image")
                    opt_start_step = gr.Slider(label="Optimization schedule", value=0.8, minimum=0.0, maximum=1, step=0.05,
                                               info="Optimization is applied after this fraction of steps. Recommended to match the attention control schedule.")

                with gr.Group():
                    enable_spl = gr.Checkbox(label="Preserve structure (SPL)", value=True)
                    with gr.Column(visible=True) as spl_options:
                        spl_area = gr.Dropdown(["Whole image", "Masked area"], value="Whole image", label="Preservation area")

                with gr.Group():
                    enable_cpl = gr.Checkbox(label="Preserve color (CPL)", value=False)
                    with gr.Column(visible=False) as cpl_options:
                        cpl_area = gr.Dropdown(["Whole image", "Masked area"], value="Whole image", label="Preservation area")

                with gr.Group(visible=False) as mask_group:
                    gr.Markdown("<center><b>Mask settings</b></center>")
                    with gr.Row():
                        mask_words_src = gr.Textbox(label="Mask words from source", value="")
                        mask_thresh_src = gr.Slider(label="Threshold (source)", value=0.5, minimum=0.0, maximum=1.0, step=0.01)
                        invert_mask_src = gr.Checkbox(label="Invert source mask", value=False)
                    with gr.Row():
                        mask_words_tgt = gr.Textbox(label="Mask words from target", value="")
                        mask_thresh_tgt = gr.Slider(label="Threshold (target)", value=0.5, minimum=0.0, maximum=1.0, step=0.01)
                        invert_mask_tgt = gr.Checkbox(label="Invert target mask", value=False)

                generate1 = gr.Button(value="Run")

                # Event handlers for dynamic visibility
                enable_spl.change(fn=lambda v: gr.update(visible=v), inputs=enable_spl, outputs=spl_options)
                enable_cpl.change(fn=lambda v: gr.update(visible=v), inputs=enable_cpl, outputs=cpl_options)

                def update_mask_visibility(spl_on, spl_a, cpl_on, cpl_a):
                    show = (spl_on and spl_a == "Masked area") or (cpl_on and cpl_a == "Masked area")
                    return gr.update(visible=show)

                for comp in [enable_spl, spl_area, enable_cpl, cpl_area]:
                    comp.change(fn=update_mask_visibility,
                                inputs=[enable_spl, spl_area, enable_cpl, cpl_area],
                                outputs=mask_group)

            with gr.Tab("Advanced"):
                with gr.Group():
                    with gr.Row():
                        random_seed = gr.Checkbox(label="Random seed", value=False)
                        seed = gr.Number(label="Seed", value=0, precision=0)
                    num_inference_steps = gr.Slider(label="Inference steps", value=15, minimum=1, maximum=50, step=1)
                    random_seed.change(fn=lambda v: gr.update(visible=not v), inputs=random_seed, outputs=seed)
                    with gr.Row():
                        structure_weight = gr.Slider(label="Structure loss weight", value=10000, minimum=0.0, maximum=10000, step=1000)
                        color_weight = gr.Slider(label="Color loss weight", value=1000, minimum=0.0, maximum=10000, step=100)
                    with gr.Row():
                        opt_iter = gr.Number(label="Optimization iteration", value=100)
                        post_opt = gr.Checkbox(label="Post-processing with loss", value=True)
                    with gr.Row():
                        guidance_s = gr.Slider(label="Source guidance scale", value=1, minimum=1, maximum=10)
                        guidance_t = gr.Slider(label="Target guidance scale", value=2, minimum=1, maximum=10)

                generate3 = gr.Button(value="Run")

    inputs1 = [img, source_prompt, target_prompt,
               guidance_s, guidance_t,
               num_inference_steps, seed,
               attn_control_steps, opt_start_step,
               enable_spl, spl_area, enable_cpl, cpl_area,
               structure_weight, color_weight, opt_iter, post_opt,
               mask_words_src, mask_thresh_src, invert_mask_src,
               mask_words_tgt, mask_thresh_tgt, invert_mask_tgt,
               random_seed]
    generate1.click(inference, inputs=inputs1, outputs=gallery_out)
    generate3.click(inference, inputs=inputs1, outputs=gallery_out)

    gr.Examples(
        [
          # image, source_prompt, target_prompt,
          # guidance_s, guidance_t, num_inference_steps, seed,
          # attn_control_steps, opt_start_step,
          # enable_spl, spl_area, enable_cpl, cpl_area,
          # structure_weight, color_weight, opt_iter, post_opt,
          # mask_words_src, mask_thresh_src, invert_mask_src,
          # mask_words_tgt, mask_thresh_tgt, invert_mask_tgt,
          # random_seed
          ["images/marshmallow.jpg", "a marshmallow on a skewer", "a marshmallow on a campfire",
           1, 2, 15, 11,
           0.5, 0.6,
           True, "Masked area", False, "Whole image",
           10000, 1000, 100, True,
           "marshmallow", 0.65, False,
           "", 0.5, False,
           False],
          ["images/cats.png", "two cats playing on a grassy field", "two cats playing on a rainy field",
           1, 2, 15, 2,
           0.6, 0,
           True, "Masked area", False, "Whole image",
           10000, 1000, 100, True,
           "cats", 0.4, False,
           "", 0.5, False,
           False],
          ["images/old_man.png", "A grayscale portrait of an older man wearing a plaid scarf", "A colorful portrait of an older man wearing a plaid scarf",
           1, 2, 15, 0,
           0.8, 0.8,
           True, "Whole image", False, "Whole image",
           10000, 1000, 100, True,
           "", 0.5, False,
           "", 0.5, False,
           False],
          ["images/woman.png", "A woman floating in a starry sky", "A woman floating in an underwater scene",
           1, 2, 15, 3,
           0.8, 0.8,
           True, "Masked area", True, "Masked area",
           10000, 500, 100, True,
           "woman", 0.4, False,
           "", 0.5, False,
           False],
        ],
        inputs1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args = parser.parse_args()
    demo.launch(share=args.share, server_port=args.port, inbrowser=True)

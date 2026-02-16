# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image


def register_attention_control(model, controller):
    class AttnProcessor():
        def __init__(self, ty = "self", place_in_unet = "up", res = 0, idx = 0):
            self.place_in_unet = place_in_unet
            # Added for rough mask perturbation
            self.ty = ty
            self.res = res
            self.idx = idx

        def __call__(self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):
            # The `Attention` class can call different attention processors / attention functions
    
            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            h = attn.heads
            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)

            if not is_cross:
                q,k,v = controller.self_attn_forward(q, k, v, attn.heads)

            attention_probs = attn.get_attention_scores(q, k, attention_mask)
            if is_cross:
                attention_probs  = controller(attention_probs , is_cross, self.place_in_unet)
            hidden_states = torch.bmm(attention_probs, v)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj   
            # hidden_states = attn.to_out[0](hidden_states, scale=scale)
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states


    def register_recr(net_, count, place_in_unet):
        for idx, m in enumerate(net_.modules()):
            # print(m.__class__.__name__)
            if m.__class__.__name__ == "Attention":
                count+=1
                m.processor = AttnProcessor( place_in_unet)
        return count
    
    def inject_block(blocks=model.unet.up_blocks, pos="up"):
        #ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py
        count = 0
        res = -1
        if pos == "mid":
            children = [blocks]
        else:
            children = blocks.children()
        for net_ in children:
            if net_.__class__.__name__ in ["CrossAttnUpBlock2D","CrossAttnDownBlock2D","UNetMidBlock2DCrossAttn"]:
                res += 1
                idx = -1
                for atn in net_.attentions:
                        if atn.__class__.__name__ == "Transformer2DModel":
                            idx += 1
                            for block in atn.transformer_blocks:
                                if block.__class__.__name__ == "BasicTransformerBlock":
                                    #self attention
                                    if block.attn1.__class__.__name__ == "Attention":
                                        block.attn1.processor = AttnProcessor(ty = "self", place_in_unet = pos, res = res, idx = idx)
                                        count += 1
                                    #cross attention
                                    if block.attn2.__class__.__name__ == "Attention":
                                        block.attn2.processor = AttnProcessor(ty = "cross", place_in_unet = pos, res = res, idx = idx)
                                        count += 1
        return count

    cross_att_count = 0

    cross_att_count += inject_block(model.unet.up_blocks, pos="up")
    cross_att_count += inject_block(model.unet.down_blocks, pos="down")
    cross_att_count += inject_block(model.unet.mid_block, pos="mid")

    # sub_nets = model.unet.named_children()
    # for net in sub_nets:
    #     if "down" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "down")
    #     elif "up" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "up")
    #     elif "mid" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def aggregate_attention(prompts, attention_store, res, from_where, is_cross, select):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def get_rough_mask(prompts, tokenizer, attention_store, res, from_where, select=0, threshold=0.5, mask_words=None):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)

    if mask_words is None:
        # inv_alphas = 1 - attention_store.alphas
        # inv_alphas = inv_alphas.squeeze().bool()
        # attention_maps = attention_maps[:,:,inv_alphas]
        mask = torch.zeros(attention_maps.shape[0], attention_maps.shape[1])
    else:
        total_word_inds = np.empty(0)
        for word in mask_words:
            word_inds = get_word_inds(prompts[1], word, tokenizer)
            total_word_inds = np.append(total_word_inds, word_inds)
        attention_maps = attention_maps[:,:,total_word_inds]
        attention_maps = attention_maps / attention_maps.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        
        mask = attention_maps > threshold
        mask = torch.any(mask, dim=2).to(dtype=attention_maps.dtype)

    return mask

# Convert an image whose type is torch.Tensor to pil Image
def tensor_to_pil_image(tensor:torch.Tensor, normalize=True, threshold=None):
    if tensor.ndim == 4:
        tensor = tensor[0]  # take the first image
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    if normalize and tensor.max() > 0:
        tensor = tensor / tensor.max()
    if threshold is not None:
        tensor = (tensor > threshold).float()
    image_array = (tensor.float() * 255).squeeze(-1).cpu().numpy().astype(np.uint8)
    return Image.fromarray(image_array)
"""
Utilities for saving GPT-2 parameters as NumPy arrays in a format suitable for llaf.
"""


import os
from typing import Tuple

import numpy as np
import torch
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block


def to_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Returns PyTorch tensors as NumPy arrays.
    """
    return tensor.detach().cpu().numpy().astype(np.float32)


def ext_attn(attn: GPT2Attention) -> Tuple[np.ndarray, ...]:
    """
    Extracts the parameters of an attention module and returns them as NumPy arrays for llaf.
    """
    w_in, b_in = to_np(attn.c_attn.weight), to_np(attn.c_attn.bias)
    wq, wk, wv = np.split(w_in, 3, axis=1)
    bq, bk, bv = np.split(b_in, 3, axis=0)

    wq = wq.reshape(attn.embed_dim, attn.num_heads, attn.head_dim).transpose(1, 0, 2)
    wk = wk.reshape(attn.embed_dim, attn.num_heads, attn.head_dim).transpose(1, 0, 2)
    wv = wv.reshape(attn.embed_dim, attn.num_heads, attn.head_dim).transpose(1, 0, 2)
    w_in = np.stack([wq, wk, wv])

    bq = bq.reshape(attn.num_heads, attn.head_dim)
    bk = bk.reshape(attn.num_heads, attn.head_dim)
    bv = bv.reshape(attn.num_heads, attn.head_dim)
    b_in = np.stack([bq, bk, bv])

    w_out, b_out = to_np(attn.c_proj.weight), to_np(attn.c_proj.bias)
    w_out = w_out.reshape(attn.num_heads, attn.head_dim, attn.embed_dim)

    return w_in, b_in, w_out, b_out


def ext_block(block: GPT2Block) -> Tuple[np.ndarray, ...]:
    """
    Extracts the parameters of a transformer block and returns them as NumPy arrays for llaf.
    """
    gamma1, beta1 = to_np(block.ln_1.weight), to_np(block.ln_1.bias)
    gamma2, beta2 = to_np(block.ln_2.weight), to_np(block.ln_2.bias)
    w1, b1 = to_np(block.mlp.c_fc.weight), to_np(block.mlp.c_fc.bias)
    w2, b2 = to_np(block.mlp.c_proj.weight), to_np(block.mlp.c_proj.bias)
    return gamma1, beta1, gamma2, beta2, *ext_attn(block.attn), w1, b1, w2, b2


def save_gpt2(name: str = 'gpt2') -> None:
    """
    Saves the parameters of GPT-2 as NumPy arrays for llaf.

    Args:
        name: GPT-2 variant to use.
            Options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'.
    """
    gpt2 = GPT2LMHeadModel.from_pretrained(name)
    tok_emb, pos_emb = to_np(gpt2.transformer.wte.weight), to_np(gpt2.transformer.wpe.weight)
    gamma, beta = to_np(gpt2.transformer.ln_f.weight), to_np(gpt2.transformer.ln_f.bias)
    w = to_np(gpt2.lm_head.weight.T)

    block_params = zip(*[ext_block(block) for block in gpt2.transformer.h])
    block_params = [np.stack(params) for params in block_params]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, name + '.npz')
    np.savez(path, tok_emb=tok_emb, pos_emb=pos_emb,
             gamma1s=block_params[0], beta1s=block_params[1],
             gamma2s=block_params[2], beta2s=block_params[3],
             w_ins=block_params[4], b_ins=block_params[5],
             w_outs=block_params[6], b_outs=block_params[7],
             w1s=block_params[8], b1s=block_params[9],
             w2s=block_params[10], b2s=block_params[11],
             gamma=gamma, beta=beta, w=w)

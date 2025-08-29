"""
Runs GPT-2 inference using llaf.
"""


import argparse
import os
import time

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import llm
from params import save_gpt2

import futhark_data

def main(text: str, name: str = 'gpt2', cnt: int = 20, bench: bool = False, dump: str | None = None) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, name + '.npz')
    if not os.path.exists(path):
        save_gpt2(name)
    params = np.load(path)

    llaf = llm.llm()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    ids = np.array(tokenizer.encode(text), dtype=np.int64)

    if dump is not None:
        print(f'Writing Futhark data files to {dump}.')
        os.makedirs(dump, exist_ok=True)
        for v in ['tok_emb', 'pos_emb', 'gamma1s', 'beta1s', 'gamma2s', 'beta2s', 'w_ins', 'b_ins', 'w_outs', 'b_outs', 'w1s', 'b1s', 'w2s', 'b2s', 'gamma', 'beta', 'w']:
            futhark_data.dump(params[v], open(f'{dump}/{v}.data', 'wb'), binary=True)
        futhark_data.dump(ids, open(f'{dump}/ids.data', 'wb'), binary=True)

    params = llaf.init(params['tok_emb'], params['pos_emb'],
                       params['gamma1s'], params['beta1s'],
                       params['gamma2s'], params['beta2s'],
                       params['w_ins'], params['b_ins'],
                       params['w_outs'], params['b_outs'],
                       params['w1s'], params['b1s'],
                       params['w2s'], params['b2s'],
                       params['gamma'], params['beta'], params['w'])
    out = llaf.gen(ids, params, cnt).get()
    print('Output:', tokenizer.decode(out.tolist()))

    if bench:
        for _ in range(5):
            _ = llaf.gen(ids, params, cnt).get()

        start = time.time()
        _ = llaf.gen(ids, params, cnt).get()
        end = time.time()
        print(f'\nllaf time: {end - start:.3f} s')

        torch.set_grad_enabled(False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        hf = GPT2LMHeadModel.from_pretrained(name).to(device).eval()
        ids = torch.tensor(ids, device=device).unsqueeze(0)
        mask = torch.ones_like(ids)

        for _ in range(5):
            _ = hf.generate(ids, attention_mask=mask,
                            max_new_tokens=cnt, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)

        start = time.time()
        _ = hf.generate(ids, attention_mask=mask,
                        max_new_tokens=cnt, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        print(f'Hugging Face time: {end - start:.4f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs GPT-2 inference using llaf.')
    parser.add_argument('text',
                        type=str,
                        help='Prompt text to feed into GPT-2.')
    parser.add_argument('--name',
                        type=str,
                        default='gpt2',
                        help='GPT-2 variant to use.',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--cnt',
                        type=int,
                        default=20,
                        help='Number of additional tokens to generate.')
    parser.add_argument('--bench',
                        action='store_true',
                        help='Flag for benchmarking llaf against Hugging Face.')
    parser.add_argument('--dump',
                        type=str,
                        default=None,
                        help='Dump Futhark-readable data files to this directory.')

    args = parser.parse_args()
    main(args.text, args.name, args.cnt, args.bench, args.dump)

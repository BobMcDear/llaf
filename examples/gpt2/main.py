"""
Runs GPT-2 inference using llaf.
"""


import argparse
import os
import time

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from params import save_gpt2

import futhark_data
import futhark_server

PARAM_NAMES = ['tok_emb', 'pos_emb', 'gamma1s', 'beta1s', 'gamma2s', 'beta2s', 'w_ins', 'b_ins', 'w_outs', 'b_outs', 'w1s', 'b1s', 'w2s', 'b2s', 'gamma', 'beta', 'w']

class LLM:
    def __init__(self, params):
        self.server = futhark_server.Server('./llm', '--cache=llm.cache')
        for v in PARAM_NAMES:
            self.server.put_value(v, params[v])
        self.server.cmd_call('init', 'params', *PARAM_NAMES)
        for v in PARAM_NAMES:
            self.server.cmd_free(v)

    def __enter__(self):
        self.server.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        return self.server.__exit(type, value, traceback)

    def gen(self, ids, cnt: int):
        self.server.put_value('ids', ids)
        self.server.put_value('cnt', np.int64(cnt))
        self.server.cmd_call('gen', 'out', 'ids', 'params', 'cnt')
        out = self.server.get_value('out')
        self.server.cmd_free('out')
        self.server.cmd_free('ids')
        self.server.cmd_free('cnt')
        return out

def main(text: str, name: str = 'gpt2', cnt: int = 20, bench: bool = False, dump: str | None = None) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, name + '.npz')
    if not os.path.exists(path):
        save_gpt2(name)
    params = np.load(path)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    ids = np.array(tokenizer.encode(text), dtype=np.int64)

    if dump is not None:
        print(f'Writing Futhark data files to {dump}.')
        os.makedirs(dump, exist_ok=True)
        for v in PARAM_NAMES:
            futhark_data.dump(params[v], open(f'{dump}/{v}.data', 'wb'), binary=True)
        futhark_data.dump(ids, open(f'{dump}/ids.data', 'wb'), binary=True)

    llaf = LLM(params)

    out = llaf.gen(ids, cnt)

    print('Output:', tokenizer.decode(out.tolist()))

    if bench:
        for _ in range(5):
            _ = llaf.gen(ids, cnt)

        start = time.time()
        _ = llaf.gen(ids, cnt)
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

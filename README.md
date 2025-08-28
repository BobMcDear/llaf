# llaf

• **[Introduction](#introduction)**<br>
• **[Installation](#installation)**<br>
• **[Usage](#usage)**<br>
• **[Examples](#examples)**<br>
• **[Performance](#performance)**<br>
• **[Training](#training)**<br>
• **[Citations](#citations)**<br>

## Introduction

llaf is a large language model (LLM) inference engine written in [Futhark](https://futhark-lang.org/), a functional language designed for efficient data-parallel array processing. Among tools intended for developing efficient GPU or multi-threaded CPU kernels, Futhark is unique in that it doesn't resemble low-level programming and is fully legible to anyone with a background in [Haskell](https://www.haskell.org/) or the [ML family](https://cs.lmu.edu/~ray/notes/introml/). Furthermore, unlike domain-specific languages (DSLs) like [Triton](https://triton-lang.org/main/index.html) or [Numba](https://numba.pydata.org/), Futhark is its own (small) language and doesn't suffer from the drawbacks of relying on a host. Another of its advantages is _size annotations_, which are of immense help when working with complex multi-dimensional arrays. Its familiar functional design coupled with its [cutting-edge performance](https://futhark-lang.org/performance.html) make it an appealing choice for implementing high-performance array computations on vector hardware. This project is a case study on how relevant it is for deep learning workloads.

## Installation

On Linux and macOS, Futhark can be easily installed via [Homebrew](https://brew.sh/): `brew install futhark`. Please refer to the [official installation guide](https://futhark.readthedocs.io/en/latest/installation.html) for more information. Then, clone this repository with `git clone https://github.com/BobMcDear/llaf` to get started.

## Usage

[`src/llm.fut`](https://github.com/BobMcDear/llaf/blob/main/src/llm.fut) contains the complete inference implementation, with two entry points exposed to the user:

* `gen`: Autoregressively generates token in a greedy fashion given an initial context.
    * Arguments:
        * `ids`: Initial context.
        * `ps`: Model parameters as a record.
        * `cnt`: Number of additional tokens to generate. If the sequence produced exceeds the maximum length during generation, the input to the model is truncated.
    * Returns: Generated sequence.
* `init`: Initializes the model state given pre-trained parameters.
    * Arguments:
        * `tok_emb`: Token embeddings.
        * `pos_emb`: Position embeddings.
        * `mask`: Causal self-attention mask.
        * `gamma1s`: Scale parameters of the first layer norm in each block.
        * `beta1s`: Shift parameters of the first layer norm in each block.
        * `gamma2s`: Scale parameters of the second layer norm in each block.
        * `beta2s`: Shift parameters of the second layer norm in each block.
        * `w_ins`: Attention QKV projection weights of each block.
        * `b_ins`: Attention QKV projection biases of each block.
        * `w_outs`: Attention output projection weights of each block.
        * `b_outs`: Attention output projection biases of each block.
        * `w1s`: Weights of the first MLP linear layer in each block
        * `b1s`: Biases of the first MLP linear layer in each block
        * `w2s`: Weights of the second MLP linear layer in each block
        * `b2s`: Biases of the second MLP linear layer in each block
        * `gamma`: Scale parameters of the final layer norm.
        * `beta`: Shift parameters of the final layer norm.
        * `w`: Vocabulary projection weights
    * Returns: Model parameters as a record.

The source code includes more details and comments.

## Examples

One of Futhark's backends is [PyOpenCL](https://documen.tician.de/pyopencl/), which conveniently translates Futhark code into PyOpenCL- and NumPy-powered Python. Using this interoperability, it's easy to run LLM inference in Python using llaf. [`examples/gpt2`](https://github.com/BobMcDear/llaf/tree/main/examples/gpt2) shows how to do so.

## Performance

Perhaps unsurprisingly, in the example above, Futhark can't keep up with PyTorch and is slower by 3-10x depending on the input size. However, it is not _unusably_ slow: It generates 500 tokens in about 30 s on an RTX 2070 GPU (vs the Hugging Face baseline of 3 s), which isn't bad given how optimized and specialized deep learning frameworks are for this type of task. Of course, there is most likely room for efficiency gains in the code; these results only pertain to a naive implementation of LLMs in Futhark, which can be improved upon with proper profiling and tuning.

## Training

Although llaf is intended for LLM inference, adapting it for training would be straightforward thanks to two key features of Futhark:

* `map`: Any function can be mapped over the leading axis of an array. In other words, we can apply `map` over a forward pass method that would normally take a single data point to handle batches of samples.
* `vjp`: Reverse-mode automatic differentiation can be achieved in Futhark using the built-in `vjp` function. Paired up with a loss function, this allows for simple and efficient gradient descent.

These two functionalities are one among several that Futhark shares with [JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html), which can be classified as a DSL and thus comes with many problems of its own.

## Citations

```bibtex
@software{The_Futhark_Hackers_Futhark,
author = {The Futhark Hackers},
title = {{Futhark}},
url = {https://github.com/diku-dk/futhark}
}
```
```bibtex
@inproceedings{henriksen2017futhark,
  title={Futhark: purely functional GPU-programming with nested parallelism and in-place array updates},
  author={Henriksen, Troels and Serup, Niels GW and Elsman, Martin and Henglein, Fritz and Oancea, Cosmin E},
  booktitle={Proceedings of the 38th ACM SIGPLAN Conference on Programming Language Design and Implementation},
  pages={556*571},
  year={2017}
}
```
```bibtex
@phdthesis{henriksen2017design,
  title={Design and implementation of the Futhark programming language},
  author={Henriksen, Troels},
  year={2017},
  school={University of Copenhagen, Faculty of Science [Department of Computer Science]}
}
```
-- LLM inference in Futhark.
-- All the relevant code is contained in this file,
-- and no imports are necessary besides the prelude.
-- There are two points of entry:
-- gen: Autoregressively generates token in a greedy fashion given an initial context.
-- init: Initializes the model state given pre-trained parameters.

-- The model's architecture is fully specified by the following hyperparameters:
-- mx_l: Maximum sequence length.
-- l: Input sequence length.
-- v: Vocabulary size.
-- b: Number of transformer blocks.
-- d: Embedding dimensionality.
-- h: Number of heads.
-- f: Dimensionality of the MLPs' hidden layers.
-- Each of these parameters is described below.
-- Note that a leading [b] axis means the ith entry belongs to the ith transformer block.
-- tok_emb: Token embeddings.
-- pos_emb: Position embeddings.
-- mask: Causal self-attention mask.
-- gamma1s: Scale parameters of the first layer norm in each block.
-- beta1s: Shift parameters of the first layer norm in each block.
-- gamma2s: Scale parameters of the second layer norm in each block.
-- beta2s: Shift parameters of the second layer norm in each block.
-- w_ins: Attention QKV projection weights of each block.
-- b_ins: Attention QKV projection biases of each block.
-- w_outs: Attention output projection weights of each block.
-- b_outs: Attention output projection biases of each block.
-- w1s: Weights of the first MLP linear layer in each block
-- b1s: Biases of the first MLP linear layer in each block
-- w2s: Weights of the second MLP linear layer in each block
-- b2s: Biases of the second MLP linear layer in each block
-- gamma: Scale parameters of the final layer norm.
-- beta: Shift parameters of the final layer norm.
-- w: Vocabulary projection weights
type Params [mx_l][v][b][d][h][f] = {
    tok_emb: [v][d]f32, pos_emb: [mx_l][d]f32, mask: [mx_l][mx_l]f32,
    gamma1s: [b][d]f32, beta1s: [b][d]f32, gamma2s: [b][d]f32, beta2s: [b][d]f32,
    w_ins: [b][3][h][d][d/h]f32, b_ins: [b][3][h][d/h]f32,
    w_outs: [b][h][d/h][d]f32, b_outs: [b][d]f32,
    w1s: [b][d][f]f32, b1s: [b][f]f32, w2s: [b][f][d]f32, b2s: [b][d]f32,
    gamma: [d]f32, beta: [d]f32, w: [d][v]f32
}

def matmul [n][m][p] (A: [n][m]f32) (B: [m][p]f32): [n][p]f32 =
    let dot_prod (a: [m]f32) (b: [m]f32): f32 =
        reduce (+) 0.0 (map2 (*) a b)
    in map (\a -> map (dot_prod a) (transpose B)) A

def dense [n][m][p] (A: [n][m]f32) (B: [m][p]f32) (c: [p]f32): [n][p]f32 =
    map (map2 (+) c) (matmul A B)

def gelu (a: f32): f32 = -- tanh approximation of GELU, used in GPT-2.
    0.5 * a * (1 + f32.tanh (0.7978845608 * (a + 0.044715 * a * a * a)))

def softmax [n] (a: [n]f32): [n]f32 = -- Operates over vectors, later mapped over matrices.
    let shifted = map ((+) (-(reduce f32.max a[0] a))) a -- Subtracts max for stability.
    let es = map f32.exp shifted
    let sum = reduce (+) 0.0 es
    in map (\e -> e / sum) es

def argmax [n] (a: [n]f32): i64 = -- Operates over vectors, later mapped over matrices.
    let update ((mx_v, mx_i): (f32, i64)) ((curr_v, curr_i): (f32, i64)) =
        if mx_v < curr_v then (curr_v, curr_i) else (mx_v, mx_i)
    let (_, mx_i) = reduce_comm update (a[0], 0) (zip a (iota n)) -- reduce_comm is better optimized.
    in mx_i

def layer_norm [l][d] (xs: [l][d]f32) (gamma: [d]f32) (beta: [d]f32): [l][d]f32 =
    let norm_row (x: [d]f32): [d]f32 =
        let mean = (reduce (+) 0.0 x) / f32.i64 d
        let var = (reduce (+) 0.0 (map (\xi -> (xi - mean) * (xi - mean)) x)) / f32.i64 d
        let std = f32.sqrt (var + 1e-5)
        in map3 (\xi gi bi -> bi + gi * (xi - mean) / std) x gamma beta
    in map norm_row xs

def mhsa [mx_l][l][d][h] (xs: [l][d]f32) (mask: [mx_l][mx_l]f32)
                         (w_in: [3][h][d][d/h]f32) (b_in: [3][h][d/h]f32)
                         (w_out: [h][d/h][d]f32) (b_out: [d]f32): [l][d]f32 =
    -- In the following, we're effectively carrying out single-headed attention
    -- but mapping it over each head to get its multi-headed version.
    -- The logic becomes much clearer if you ignore the [h] axes and every outermost map.
    let s = f32.sqrt (f32.i64 d / f32.i64 h) -- Scale factor.
    let qs = map2 (dense xs) w_in[0] b_in[0]
    let ks = map2 (dense xs) w_in[1] b_in[1]
    let vs = map2 (dense xs) w_in[2] b_in[2]
    let raw_att = map2 (\q k -> matmul q (transpose k)) qs ks |> map (map (map (\a -> a / s)))
    let att = map (\head -> map2 (map2 (+)) head mask[:l, :l] |> map softmax) raw_att
    let conts = map2 matmul att vs |> transpose |> map flatten
    in dense conts (flatten w_out) b_out

def mlp [l][d][f] (xs: [l][d]f32) (w1: [d][f]f32) (b1: [f]f32) (w2: [f][d]f32) (b2: [d]f32): [l][d]f32 =
    let z = dense xs w1 b1 |> map (map gelu)
    in dense z w2 b2

def transformer [mx_l][l][v][b][d][h][f] (ids: [l]i64) (ps: Params [mx_l][v][b][d][h][f]): [l][v]f32 =
    let block (xs: [l][d]f32) (i: i64): [l][d]f32 = -- Passes the input through the ith block.
        let ln1 = layer_norm xs ps.gamma1s[i] ps.beta1s[i]
        let y1 = map2 (map2 (+)) xs (mhsa ln1 ps.mask ps.w_ins[i] ps.b_ins[i] ps.w_outs[i] ps.b_outs[i])
        let ln2 = layer_norm y1 ps.gamma2s[i] ps.beta2s[i]
        in map2 (map2 (+)) y1 (mlp ln2 ps.w1s[i] ps.b1s[i] ps.w2s[i] ps.b2s[i])
    let xs = map2 (map2 (+)) (map (\id -> ps.tok_emb[i64.min (v-1) id]) ids) ps.pos_emb[:l]
    let ys = foldl block xs (iota b)
    in matmul (layer_norm ys ps.gamma ps.beta) ps.w

entry gen [mx_l][l][v][b][d][h][f] (ids: [l]i64) (ps: Params [mx_l][v][b][d][h][f]) (cnt: i64): []i64 =
    let gen_token [n] (ids: [n]i64): [n+1]i64 =
        let ctx = drop (n - (i64.min mx_l n)) ids -- Truncates input if too long.
        let ys = transformer ctx ps
        in ids ++ [last ys |> argmax]
    in loop ids for _i < cnt do gen_token ids

entry init [mx_l][v][b][d][h][f] (tok_emb: [v][d]f32) (pos_emb: [mx_l][d]f32)
                                 (gamma1s: [b][d]f32) (beta1s: [b][d]f32) (gamma2s: [b][d]f32) (beta2s: [b][d]f32)
                                 (w_ins: [b][3][h][d][d/h]f32) (b_ins: [b][3][h][d/h]f32)
                                 (w_outs: [b][h][d/h][d]f32)   (b_outs: [b][d]f32)
                                 (w1s: [b][d][f]f32) (b1s: [b][f]f32) (w2s: [b][f][d]f32) (b2s: [b][d]f32)
                                 (gamma: [d]f32) (beta: [d]f32) (w: [d][v]f32): Params [mx_l][v][b][d][h][f] =
    {tok_emb=tok_emb, pos_emb=pos_emb, mask=tabulate_2d mx_l mx_l (\i j -> if j > i then -f32.inf else 0.0),
     gamma1s=gamma1s, beta1s=beta1s, gamma2s=gamma2s, beta2s=beta2s,
     w_ins=w_ins, b_ins=b_ins, w_outs=w_outs, b_outs=b_outs,
     w1s=w1s, b1s=b1s, w2s=w2s, b2s=b2s,
     gamma=gamma, beta=beta, w=w}

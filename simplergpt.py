import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, n_head=12):
    x = linear(x, **c_attn)
    causal_mask = (1 - np.tri(x.shape[0])) * -1e10
    q, k, v = np.array(np.split(x, 3, axis=-1))
    q_heads = list(map(lambda x: np.split(x, n_head, axis=-1), q))
    k_heads = list(map(lambda x: np.split(x, n_head, axis=-1), k))
    v_heads = list(map(lambda x: np.split(x, n_head, axis=-1), v))

    out_heads = list(map(lambda i: attention(q_heads[i], k_heads[i], v_heads[i], causal_mask), range(n_head)))
    
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]

    out = linear(np.hstack(out_heads), **c_proj)
    return out

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    inputs = encoder.encode(prompt)
    assert len(inputs) + n_tokens_to_generate < hparams["n_ctx"]

    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating"): 
        x = params['wte'][inputs] + params['wpe'][range(len(inputs))]
        for block in params['blocks']:
            mha_prenorm = layer_norm(x, **block['ln_1'])
            x = x + mha(mha_prenorm, **block['attn'])
            ffn_prenorm = layer_norm(x, **block['ln_2'])
            x = x + ffn(ffn_prenorm, **block['mlp'])
        logits = layer_norm(x, **params['ln_f']) @ params['wte'].T

        next_id = np.argmax(logits[-1])
        inputs = np.append(inputs, [next_id])
    output_ids = list(inputs[len(inputs) - n_tokens_to_generate :])
    return encoder.decode(output_ids)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

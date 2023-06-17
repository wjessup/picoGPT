import timeit
import numpy as np
from utils import load_encoder_hparams_and_params
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

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
    soft = softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) 
    return soft @ v

def main(prompt: str, n_tokens_to_generate: int = 10):

    print("Prompt = ", prompt)
    inputs = encoder.encode(prompt)
    assert len(inputs) + n_tokens_to_generate < hparams["n_ctx"]

    for _ in range(n_tokens_to_generate): 
        x = params['wte'][inputs] + params['wpe'][range(len(inputs))]  
        causal_mask = (1 - np.tri(len(inputs))) * -1e10
        for block in params['blocks']:
            _residual = x
            x = linear(layer_norm(x, **block['ln_1']), **block['attn']['c_attn'])
            
            split_x = np.split(x, 3, axis=-1)
            qkv_heads = list(map(lambda x: np.split(x, hparams['n_head'], axis=-1), split_x)) # size: [3, 12, input_len, 64]
            out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)] # size: [12, input_len, 64]

            x = linear(np.hstack(out_heads), **block['attn']['c_proj'])
            x = _residual + x
        
            x = x + ffn(layer_norm(x, **block['ln_2']), **block['mlp'])
        logits = layer_norm(x, **params['ln_f']) @ params['wte'].T

        next_id = np.argmax(logits[-1])
        inputs = np.append(inputs, [next_id])
        print(encoder.decode([next_id]), end="", flush=True)
        
    output_ids = list(inputs[len(inputs) - n_tokens_to_generate :])
    print("\n all done!")
    return encoder.decode(output_ids)

if __name__ == "__main__":
    import fire
    prompt = "not all heroes wear capes"
    execution_time = timeit.timeit(lambda: fire.Fire(main(prompt)), number=1)
    print(f"Execution time: {execution_time} seconds")

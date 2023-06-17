import numpy as np
import timeit
from utils import load_encoder_hparams_and_params
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

def count_padding(arr):
    return np.argmax(np.array(arr[::-1]) != 0)

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

def gpt2(inputs, max_length, wte, wpe, blocks, ln_f):
    print()
    print("inputs ids  = ", inputs)

    pad_len = count_padding(inputs)
    print("pad length = ", pad_len)

    inputs_length = max_length - pad_len
    
    causal_mask = (1 - np.tri(inputs_length)) * -1e10
    print("original mask = \n", causal_mask)

    mask = np.pad(causal_mask, ((0, pad_len), (0, pad_len)), mode='constant', constant_values=-1e10)
    print("padded mask =\n ", mask)

    x = wte[inputs] + wpe[range(len(inputs))] 

    for block in blocks:
        _residual = x
 
        mha_prenorm = layer_norm(x, **block['ln_1'])
        x = linear(mha_prenorm, **block['attn']['c_attn'])

        split_x = np.split(x, 3, axis=-1)
        qkv_heads = list(map(lambda x: np.split(x, hparams['n_head'], axis=-1), split_x))
        out_heads = [attention(q, k, v, mask) for q, k, v in zip(*qkv_heads)]
        out = linear(np.hstack(out_heads), **block['attn']['c_proj'])

        x = _residual + out
        
        ffn_prenorm = layer_norm(x, **block['ln_2'])
        x = x + ffn(ffn_prenorm, **block['mlp'])

    logits = layer_norm(x, **ln_f) @ wte.T
    
    logit_index = 1 + pad_len

    next_id = np.argmax(logits[-logit_index])
    print("generated next id = ", next_id)
    print("generated next token = ", encoder.decode([next_id]))

    if pad_len > 0:
        inputs = inputs[:-pad_len]

    full = np.concatenate([inputs, [next_id], np.zeros(pad_len)])
    print("generated sequence = ", full)
    print("generated sequence = ", encoder.decode(full))
    
    return full


            
def main(n_tokens_to_generate: int = 10, model_size: str = "124M", models_dir: str = "models"):
    
    
    #strings = ["not all heroes wear capes"] 
    strings = ["not all heroes wear capes", "all tacos are tasty and del ici ous"] 

    # Tokenize and add padding to all prompts
    input_ids = [encoder.encode(a) for a in strings]    
    max_length = max(len(lst) for lst in input_ids)
    padded_inputs = [np.pad(lst, (0, max_length - len(lst)), mode='constant') for lst in input_ids]
    
    # Generate tokens, one at a time, for each sequence in the batch
    for _ in range(n_tokens_to_generate):
        outputs = np.apply_along_axis(lambda x: gpt2(x, max_length, **params), 1, padded_inputs)
        print("output datatype = ", outputs.dtype)
        print("full = ", outputs.astype(int))
        padded_inputs = outputs.astype(int)

        max_length += 1
        
    output_ids = outputs

    print(output_ids)
    output_text = encoder.decode(output_ids)
    print(output_text)
    return output_text


if __name__ == "__main__":
    import fire
    
    execution_time = timeit.timeit(lambda: fire.Fire(main), number=1)
    print(f"Execution time: {execution_time} seconds")

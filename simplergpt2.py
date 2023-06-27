import timeit
import numpy as np
#from jax import grad, jit, vmap
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

def mlp(x, c_fc, c_proj): 
    l1 = linear(x, **c_fc) # increase to 4x embed -> [input x 3072]
    g = gelu(l1) # magic
    return linear(g, **c_proj) # decrease back to embed -> [input x 768]

def attention(q, k, v, mask):
    soft = softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) 
    return soft @ v

def main(prompt: str, n_tokens_to_generate: int = 10):

    print("Prompt = ", prompt)
    inputs = np.array(encoder.encode(prompt))
    assert len(inputs) + n_tokens_to_generate < hparams["n_ctx"]

    for _ in range(n_tokens_to_generate): 
        residual_stream = params['wte'][inputs] + params['wpe'][range(len(inputs))]  
        causal_mask = (1 - np.tri(len(inputs))) * -1e10
        for block in params['blocks']:
            
            ln1 = layer_norm(residual_stream, **block['ln_1'])

            # composition of linear maps is just another linear map.
            # this is where you build 3 matricies, with enough to split into Q, K, V, each of 768. 
            qkv = linear(ln1, **block['attn']['c_attn']) # [n_seq, n_embd] @ [n_embd, 3*n_embd] -> [n_seq, 3*n_embd]
    
            split_x = np.split(qkv, 3, axis=-1)
            qkv_heads = list(map(lambda x: np.split(x, hparams['n_head'], axis=-1), split_x))
            out_heads = [softmax(q @ k.T / np.sqrt(q.shape[-1]) + causal_mask) @ v for q, k, v in zip(*qkv_heads)]
            attn_scores = linear(np.hstack(out_heads), **block['attn']['c_proj'])
            
            residual_stream = residual_stream + attn_scores

            ln2 = layer_norm(residual_stream, **block['ln_2'])

            mlp_out = mlp(ln2, **block['mlp'])

            residual_stream = residual_stream + mlp_out

        logits = layer_norm(residual_stream[-1], **params['ln_f']) @ params['wte'].T # slice X here to save some computations
        next_id = np.argmax(logits)
        inputs = np.append(inputs, next_id)
        #print(inputs)
        print(encoder.decode([int(next_id)]), end="", flush=True)
        
    output_ids = np.array(inputs[len(inputs) - n_tokens_to_generate :])
    print("\n all done!")
    return encoder.decode(output_ids)

if __name__ == "__main__":
    import fire
    prompt = "not all heroes wear capes"
    #main(prompt)
    execution_time = timeit.timeit(lambda: fire.Fire(main(prompt)), number=10)
    print(f"Execution time: {execution_time} seconds")

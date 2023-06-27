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

    if x.ndim == 0:
        mean = np.mean(x,  keepdims=True)
        variance = np.var(x,  keepdims=True)
    else:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

    #mean = np.mean(x, axis=-1, keepdims=True)
    #variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def mlp(x, c_fc, c_proj): 
    l1 = linear(x, **c_fc) # increase to 4x embed -> [input x 3072]
    g = gelu(l1) # magic
    return linear(g, **c_proj) # decrease back to embed -> [input x 768]

def attention(q, k, v, mask):
    z = q @ k.T
    soft = softmax( z / np.sqrt(q.shape[-1]) + mask) 
    return soft @ v

def attention_log(q, k, v, mask):
    z = q @ k.T    
    soft = softmax( z / np.sqrt(q.shape[-1]) ) 
    y = soft @ v
    return y

def main(prompt: str, n_tokens_to_generate: int = 10):

    print("Prompt = ", prompt)
    inputs = np.array(encoder.encode(prompt))
    assert len(inputs) + n_tokens_to_generate < hparams["n_ctx"]

    
    cache_k = [None] * len(params['blocks'])
    cache_v = [None] * len(params['blocks'])

    causal_mask = (1 - np.tri(len(inputs))) * -1e10

    for _ in range(n_tokens_to_generate): 
        residual_stream = params['wte'][inputs] + params['wpe'][range(len(inputs))]  
         #potentially move this outside the loop. the mask is only necessary on the first pass
        for block_id, block in enumerate(params['blocks']):
            
            # on the first pass we process all tokens to create the cache
            # on subsequent passes we only process the last token and append the results to the cache
            if _ != 0 and block_id == 0:
                residual_stream = residual_stream[-1]

            ln1 = layer_norm(residual_stream, **block['ln_1'])

            qkv = linear(ln1, **block['attn']['c_attn']) # [n_seq, n_embd] @ [n_embd, 3*n_embd] -> [n_seq, 3*n_embd]
    
            split_x = np.split(qkv, 3, axis=-1)
            qkv_heads = np.array(list(map(lambda x: np.split(x, hparams['n_head'], axis=-1), split_x))) 

            #print(qkv_heads.shape) # (3, 12, 6, 64)

            if _ == 0:
                cache_k[block_id] = qkv_heads[1] 
                cache_v[block_id] = qkv_heads[2]
                out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
                attn_scores = linear(np.hstack(out_heads), **block['attn']['c_proj'])
            

            else:
                k = qkv_heads[1] # (12, 64)
                k_reshaped = k.reshape(12, 1, 64)
                cache_k[block_id] = np.concatenate((cache_k[block_id], k_reshaped), axis=1)

                v = qkv_heads[2] # (12, 64)
                v_reshaped = v.reshape(12, 1, 64)
                cache_v[block_id] = np.concatenate((cache_v[block_id], v_reshaped), axis=1)

                out_heads = []
                for head_id, (q, k, v) in enumerate(zip(*qkv_heads)):
                    #print("head ", head_id)
                    #print(v.shape)
                    k = cache_k[block_id][head_id]
                    v = cache_v[block_id][head_id]
                    #print(v.shape)
                    out = attention_log(q, k, v, causal_mask)
                    out_heads.append(out)


            attn_scores = linear(np.hstack(out_heads), **block['attn']['c_proj'])
            
            residual_stream = residual_stream + attn_scores

            ln2 = layer_norm(residual_stream, **block['ln_2'])
            mlp_out = mlp(ln2, **block['mlp'])                
        
            residual_stream = residual_stream + mlp_out


        #print("token = ", _)
        #print(residual_stream.shape)
        if _ == 0:
            logits = layer_norm(residual_stream[-1], **params['ln_f']) @ params['wte'].T # slice X here to save some computations
        else:
            logits = layer_norm(residual_stream, **params['ln_f']) @ params['wte'].T # slice X here to save some computations

        next_id = np.argmax(logits)
        inputs = np.append(inputs, next_id)
        print(encoder.decode([int(next_id)]), end="", flush=True)
        
    output_ids = list(inputs[len(inputs) - n_tokens_to_generate :])
    print("\n all done!")
    return encoder.decode(output_ids)

if __name__ == "__main__":
    import fire
    prompt = "not all heroes wear capes"
    #main(prompt)
    execution_time = timeit.timeit(lambda: fire.Fire(main(prompt)), number=10)
    print(f"Execution time: {execution_time} seconds")

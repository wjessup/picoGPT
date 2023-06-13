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
    temp = q @ k.T / np.sqrt(q.shape[-1])
    masked = temp + mask
    soft = softmax(masked)
    #print(soft)
    out = soft @ v
    return out
    #return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, mask, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    
    
    out_heads = [attention(q, k, v, mask) for q, k, v in zip(*qkv_heads)]
    out = linear(np.hstack(out_heads), **c_proj)
    return out

def transformer_block(x, pad_len, mlp, attn, ln_1, ln_2, n_head):
    norm1 = layer_norm(x, **ln_1)
    x = x + mha(norm1, pad_len, **attn, n_head=n_head)
    norm2 = layer_norm(x, **ln_2)
    x = x + ffn(norm2, **mlp)
    return x

def gpt2(inputs, max_length, wte, wpe, blocks, ln_f, n_head):
    print()
    print("inputs ids  = ", inputs)

    pad_len = count_padding(inputs)
    print("pad length = ", pad_len)

    inputs_length = max_length - pad_len

    x = wte[inputs] + wpe[range(len(inputs))] # => [[5x768], [10x768]]
    
    causal_mask = (1 - np.tri(inputs_length)) * -1e10
    print("casual mask = \n", causal_mask)
    new_causal_mask = np.pad(causal_mask, ((0,pad_len), (0,pad_len)), mode='constant', constant_values=-1e10)

    print("new mask =\n ", new_causal_mask)


    for block in blocks:
        blockreturn = transformer_block(x, new_causal_mask, **block, n_head=n_head)
        x = blockreturn
    x = layer_norm(x, **ln_f) 
    x = x @ wte.T

    logits = x
    next_id = np.argmax(logits[-1])
    print("generated next id = ", next_id)
    print("generated next token = ", encoder.decode([next_id]))
    full = np.concatenate([inputs, [next_id]])
    print("generated sequence = ", encoder.decode(full))
    
    return next_id

def generate(inputs, max_length, params, n_head, n_tokens_to_generate):  
    _max_length = max_length  
    from tqdm import tqdm
    for id in tqdm(range(n_tokens_to_generate), "generating"):
    
        next_ids = np.apply_along_axis(lambda x: gpt2(x, _max_length, **params,n_head=n_head), 1, inputs)
        #logits = gpt2(inputs[0], **params, n_head=n_head)
        print("next_ids = ", next_ids)

        next_ids = [[l] for l in next_ids]
        print("next_ids = ", next_ids)

        inputs = np.concatenate([inputs, next_ids], axis=-1)
        print("\n\n concated inputs = ", inputs)
        
        print()
        _max_length += 1
       
    return list(inputs[len(inputs) - n_tokens_to_generate :])
            
def main(n_tokens_to_generate: int = 10, model_size: str = "124M", models_dir: str = "models"):
    
    
    #strings = ["not all heroes wear capes"] #this works properly
    strings = ["not all heroes wear capes", "all tacos are tasty and del iciou s"] #this does not.
    input_ids = [encoder.encode(a) for a in strings]
    print(input_ids)
    max_length = max(len(lst) for lst in input_ids)
    print("longest = ", max_length)

    padded_inputs = [np.pad(lst, (0, max_length - len(lst)), mode='constant') for lst in input_ids]

    #assert len(inputs) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(padded_inputs, max_length, params, hparams["n_head"], n_tokens_to_generate)
    print(output_ids)
    output_text = encoder.decode(output_ids)
    print(output_text)
    return output_text

def count_padding(d):
    e = [e for e in d]
    e.reverse()
    count = 0
    for i in e:
        if (i == 0):
            count += 1
        else:
            break
    return count

main()

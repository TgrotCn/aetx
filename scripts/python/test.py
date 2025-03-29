import struct
import json

import torch

def permute_reverse(w, heads, rotary_dim):
    head_dim = w.shape[0] // heads
    assert rotary_dim <= head_dim
    w = torch.unflatten(w, 0, (-1, head_dim))

    # wr is the rotary part, wk is the part kept unrotated
    wr = w[:, :rotary_dim]
    wk = w[:, rotary_dim:]

    # switch wr from outputting two rotary_dim/2 chunks to outputting values interleaved
    wr = torch.unflatten(wr, 1, (2, -1))
    wr = wr.transpose(1, 2)
    wr = wr.flatten(1, 2)
    # assemble the heads back

    w = torch.cat([wr, wk], dim=1)
    return torch.flatten(w, 0, 1)

w = torch.randn((8, 4))
print(w)
new_w = permute_reverse(w, 2, 4)
print(new_w)
print(w == new_w)

# Example usage:
# in_file = "./DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors"
# out_file = "./DeepSeek-R1-Distill-Qwen-1.5B/model_resaved.safetensors"
# resave_safetensors(in_file, out_file)
# print("Re-saved safetensors file to", out_file)

# Example usage:
# dims = read_safetensors_dims("./DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors")
# print("Dimensions:", dims)

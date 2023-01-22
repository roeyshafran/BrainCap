import torch
from sentence_transformers import SentenceTransformer, util
from glob import glob
import os


@torch.no_grad()
def calculate_semantic_similarity(generated_caption, real_caption, device):
    sentence_model = SentenceTransformer('all-mpnet-base-v2').to(device)
    embed_generated = sentence_model.encode(generated_caption, convert_to_tensor=True)
    embed_real_caption = sentence_model.encode(real_caption, convert_to_tensor=True)
      
    return torch.diagonal(util.pytorch_cos_sim(embed_generated, embed_real_caption))

def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def evaluate_mindvis_to_clip(path, caption_prefix='Sample_', clip_caption_prefix='Sample_Clip_', device='cpu', return_k_best=5):
    num_of_files = len(glob(os.path.join(path, caption_prefix,'*.txt')))
    captions = []
    clip_captions = []

    for i in range(num_of_files):
        caption_path = caption_prefix + str(i)
        clip_caption_path = clip_caption_prefix + str(i)

        with open(caption_path, 'rb') as f:
            captions.append(f.read())
        with open(clip_caption_path, 'rb') as f:
            clip_captions.append(f.read())

    semantic_accuracy = calculate_semantic_similarity(clip_captions, captions, device)
    mean_accuracy = torch.mean(semantic_accuracy)
    if return_k_best:
        topk_values, topk_idx = torch.topk(semantic_accuracy, return_k_best, dim=0)
        return mean_accuracy, topk_values, captions[topk_idx]
    else:
        return mean_accuracy


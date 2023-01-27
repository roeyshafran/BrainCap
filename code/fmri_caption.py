#%%
import sys
sys.path.append(r'./Mind_Vis_utils/')

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset import create_BOLD5000_dataset
from Mind_Vis_utils.dc_ldm.ldm_for_fmri import create_model_from_config as mindvis_create_model_from_config
#%%
# Constants
# TODO: Move constants to config file
TOP_K = 1000
TOP_P = 0.95
MAX_CAPTION_LEN = 100


class MLP(nn.Module):
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        """
           Taken from the ClipCap paper implementation:
               - https://arxiv.org/abs/2111.09734
               - https://github.com/rmokady/CLIP_prefix_caption 
            MLP mapping network
        Args:
            sizes (list): MLP layers sizes
            bias (bool, optional): Defaults to True.
            act (_type_, optional): Activation function to use. Defaults to nn.Tanh.
        """
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
            if i < len(sizes) - 2: # No activation on last layer
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MLP_dropout(nn.Module):
    """An identical implementaion of MLP class with drop out layers.
    """
    def __init__(self, sizes, dropout=0.3, bias=True, act=nn.Tanh):
        super(MLP_dropout, self).__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
            layers.append(nn.Dropout(p=dropout))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GPTCaptionModel(nn.Module):
    def __init__(self, prefix_length, prefix_size, projection_sizes, act=nn.Tanh, use_dropout=False):
        """Implementation of the decoder class.

        Args:
            prefix_length (int): length of the fmri prefix. Equel to the numer of patches in the fMRI scan.
            prefix_size (int): embedding dimension of the fmri encoder
            projection_sizes (list): list of the MLP layers sizes
            act (_type_, optional): Activation function. Defaults to nn.Tanh.
            use_dropout (bool, optional): If True uses MLP_dropout and not the MLP class. Defaults to False.
        """
        super(GPTCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.prefix_size = prefix_size
        self.projection_sizes = projection_sizes
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self._stop_token_index = self.tokenizer.encode('.')[0]

        assert projection_sizes[0] == self.prefix_size
        self.projection_sizes.append(self.gpt_embedding_size)
        if use_dropout:
            self.embedding_space_projection = MLP_dropout(self.projection_sizes, dropout=0.3, bias=True, act=act)    
        else:
            self.embedding_space_projection = MLP(self.projection_sizes, bias=True, act=act)

    def forward(self, tokens, fmri_prefix, mask=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        #prefix_projections = self.embedding_space_projection(fmri_prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        prefix_projections = self.embedding_space_projection(fmri_prefix)
        # concatenate the fmri prefix (fmri encoder output) and the real caption embedding to calcaulte the decoder output
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        mask_cat = torch.cat((torch.ones_like(prefix_projections[:, :, -1]), mask), dim=1)
        return self.gpt(inputs_embeds=embedding_cat, attention_mask=mask_cat)

    def _gpt_next_token(self, embedding, device):
        # given an embedding (fmri prefix + work embeddings) estimate the next word in the sentences
        gpt_output = self.gpt(inputs_embeds=embedding)
        next_token_logits = gpt_output['logits'][:, -1, :]

        # Nucleus sampling of the next token
        filtered_p = top_k_top_p_filtering(next_token_logits, top_k=TOP_K, top_p=TOP_P, device=device)
        next_token = torch.multinomial(filtered_p, num_samples=1)
        next_token_embed = self.gpt.transformer.wte(next_token)
        return next_token, next_token_embed

    @torch.no_grad()
    def generate_caption(self, fmri_prefix, device):
        generated_text_list = []
        for prefix in fmri_prefix:
            # Project fmri encoder embedding to GPT latent space
            gpt_embedding = self.embedding_space_projection(prefix)
            gpt_embedding = gpt_embedding.view(-1, self.prefix_length, self.gpt_embedding_size)

            tokens = torch.tensor([[self.tokenizer.bos_token_id]])
            # Estimate next token until maximal caption length is reached or a '.' token is reached.
            for i in range(MAX_CAPTION_LEN):
                next_token, next_token_embed = self._gpt_next_token(gpt_embedding, device)
                next_token, next_token_embed, tokens = next_token.to(device), next_token_embed.to(device), tokens.to(device)
                tokens = torch.cat((tokens, next_token), dim=1)
                gpt_embedding = gpt_embedding.to(device)
                gpt_embedding = torch.cat((gpt_embedding, next_token_embed), dim=1)

                if self._stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = self.tokenizer.decode(output_list)
            generated_text_list.append(output_text)

        return generated_text_list

    def parameters(self, recurse: bool = True):
        return self.embedding_space_projection.parameters()

    def train(self, mode: bool = True):
        super(GPTCaptionModel, self).train(mode)
        # Freeze GPT2 model
        self.gpt.eval()
        return self


# TODO: Check what is the best way to use torch.load map_location parameter
def create_fmri_encoder_from_pretrained(path_pretrain_mbm_metafile, num_voxels, global_pool=False, feature_extraction=True):
    pretrain_mbm_metafile = torch.load(path_pretrain_mbm_metafile, map_location='cpu')
    pretrain_mbm_metafile = pretrain_mbm_metafile['metafile']

    # fmri_encoder object
    model = mindvis_create_model_from_config(pretrain_mbm_metafile['config'], num_voxels, global_pool)
    model.load_checkpoint(pretrain_mbm_metafile['model'])

    set_parameter_requires_grad(model, feature_extraction=feature_extraction)
    
    return model

# Taken from:
# https://sachinruk.github.io/blog/pytorch/huggingface/2021/12/28/vit-to-gpt2-encoder-decoder-model.html#Training-Module-(PyTorch-Lightning)
def top_k_top_p_filtering(
        next_token_logits: torch.FloatTensor,
        top_k = None,
        top_p = None,
        device = "cpu",
):
    p, largest_p_idx = F.softmax(next_token_logits, dim=-1).topk(top_k, dim=-1)
    cumulative_p = p.cumsum(dim=-1)
    threshold_repeated = top_p + torch.zeros((len(p), 1)).to(device)
    idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k - 1).squeeze()
    cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
    censored_p = (cumulative_p <= cutoffs[:, None]) * p
    renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)

    final_p = torch.zeros_like(next_token_logits)
    row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1, top_k).to(device)
    final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)

    return final_p

def set_parameter_requires_grad(model, feature_extraction=True):
  if feature_extraction:
    model.requires_grad_(False)
  else:
    model.requires_grad_(True)

def remove_duplicates_from_dataset(dataloader, num_voxels, device):
    prev_seen = {
            'caption': np.array([]),
            'image': torch.tensor([]).to(device),
            'fmri': torch.tensor([]).to(device)
        }
    for batch in dataloader:
        #batch['fmri'] = batch['fmri'].to(device)
        #batch['image'] = batch['image'].to(device)
        remove_duplicates_in_batch(batch) # in-place
        remove_previously_seen_fmri(batch, prev_seen, device) # in-place
        
        try:
            prev_seen['caption'] = np.concatenate((prev_seen['caption'], batch['caption']))
        except:
            prev_seen['caption'] = np.concatenate((prev_seen['caption'], [batch['caption']]))
        prev_seen['image'] = torch.cat((prev_seen['image'], batch['image'].to(device)), dim=0)
        prev_seen['fmri'] = torch.cat((prev_seen['fmri'], batch['fmri'].to(device)), dim=0)

    #return [dict(zip(prev_seen,t)) for t in zip(*prev_seen.values())]
    return BOLD5000_dataset(prev_seen['fmri'].cpu(), prev_seen['caption'].cpu(), prev_seen['image'].cpu(), identity, identity, num_voxels)

def define_GPTCaption_model(encoder, trial=None, projection_sizes=None, use_dropout=False):
    if trial:
        # TODO: Add Optuna support. If trial is used, override projection_sizes with Optuna suggestion
        num_layers = trial.suggest_int("num_projection_layers", 1, 5)
        projection_sizes = [encoder.embed_dim]*num_layers
    gpt_decoder = GPTCaptionModel(encoder.num_patches, encoder.embed_dim, projection_sizes, use_dropout=use_dropout)
    set_parameter_requires_grad(gpt_decoder.embedding_space_projection, feature_extraction=False)
    set_parameter_requires_grad(gpt_decoder.gpt, feature_extraction=True)
    #set_parameter_requires_grad(gpt_decoder.tokenizer, feature_extraction=True)
    #for param in gpt_decoder.tokenizer.parameters():
    #  param.requires_grad = False

    return gpt_decoder

def main():
    path_encoder = r"/databases/roeyshafran/BrainCap/pretrains/pretrain_metafile.pth"
    path_BOLD = r"/databases/roeyshafran/BrainCap/data/BOLD5000/CSI1_dataset.pth"

    device = 'cuda:0'

    # create BOLD5000 dataset
    print("Loading BOLD5000 Dataset")
    BOLD_dataset = torch.load(path_BOLD)
    train, test = BOLD_dataset['train'], BOLD_dataset['test']

    train_dl = DataLoader(train, batch_size=5)
    s1 = iter(train_dl).next()

    batch_fmri = s1['fmri'].to(device)
    batch_caption = s1['caption']



    print("Creating Encoder")
    encoder = create_fmri_encoder_from_pretrained(path_encoder, train.num_voxels).to(device)
    print("Creating Decoder")
    decoder = GPTCaptionModel(encoder.num_patches, encoder.embed_dim, [encoder.embed_dim]).to(device)

    encoded_fmri = encoder.forward(batch_fmri)
    #batch_tokens = decoder.tokenizer(batch_caption, return_tensors="pt", padding=True)
    generated_text = decoder.generate_caption(encoded_fmri.to(device), device)
    #outputs = decoder.forward(batch_tokens['input_ids'], encoded_fmri, batch_tokens['attention_mask'])
    pass
if __name__ == "__main__":
    main()
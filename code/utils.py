import sys
sys.path.append(r'/home1/roeyshafran/BrainCap/Mind-Cap/code/Mind_Vis_utils/')

from fmri_caption import GPTCaptionModel, create_fmri_encoder_from_pretrained,top_k_top_p_filtering, set_parameter_requires_grad
from dataset import BOLD5000_dataset, identity
from dataset import create_BOLD5000_dataset
from torch.utils.data import DataLoader, Subset
import torch
import torch.optim as optim
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import heapq
from datetime import datetime
from glob import glob
import os


@torch.no_grad()
def calculate_semantic_similarity(generated_caption, real_caption, device):
    sentence_model = SentenceTransformer('all-mpnet-base-v2').to(device)
    embed_generated = sentence_model.encode(generated_caption, convert_to_tensor=True)
    embed_real_caption = sentence_model.encode(real_caption, convert_to_tensor=True)
      
    return torch.diagonal(util.pytorch_cos_sim(embed_generated, embed_real_caption))


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

def print_batch(batch, fontsize=5, num_of_columns=5, caption_as_title=False):
    N = int(np.ceil(np.sqrt(len(batch))))
    num_of_plots = len(batch)
    num_of_rows = num_of_plots // num_of_columns
    fig, axs = plt.subplots(num_of_rows, num_of_columns)
    if hasattr(axs, '__iter__'):
        axs = axs.flatten()
    else:
        axs = [axs]
    for idx, ax in enumerate(axs):
        try:
            ax.imshow(batch[idx]['image'])
            str_to_show = f"({idx}) Generated: {batch[idx]['caption']}\n   Real: {batch[idx]['real_caption']}  Accuracy: {batch[idx]['accuracy']}"
            if caption_as_title:
                ax.set_title(str_to_show, fontsize=fontsize)
            else:
                print(str_to_show)
        except IndexError:
            pass
        ax.set_xticks([],[])
        ax.set_yticks([],[])
      
    fig.tight_layout()



@torch.no_grad()
def calculate_accuracy_on_test(encoder, decoder, dataloader, device, threshhold=0.5, return_best_batch=False):
    running_accuracy = 0
    above_threshhold_count = 0
    best_accuracy = 0
    for batch in dataloader:
        fmri_prefix = encoder.forward(batch['fmri'].to(device))
        generated_caption = decoder.generate_caption(fmri_prefix, device)

        accuracy_tensor = calculate_semantic_similarity(generated_caption, batch['caption'], device)
        above_threshhold_count += torch.numel(accuracy_tensor[accuracy_tensor >= threshhold])
        accuracy = torch.mean(accuracy_tensor).item()
        running_accuracy += accuracy

        if return_best_batch and (accuracy > best_accuracy):
            best_accuracy = accuracy
            fields = ['accuracy', 'caption', 'real_caption', 'image']
            best_batch = [dict(zip(fields, t)) for t in zip(accuracy_tensor, generated_caption, batch['caption'], batch['image'])]
            #best_batch = list(zip(generated_caption, batch['caption'], batch['image']))

    accuracy = (running_accuracy / len(dataloader), above_threshhold_count / (len(dataloader)*dataloader.batch_size)) 
    if return_best_batch:
        return accuracy, best_batch
    else:
        return accuracy
        
def get_k_best(encoder, decoder, dataloader, k, device):
    k_best = []
    for batch in dataloader:
        remove_duplicates_in_batch(batch) # in-place
        remove_previously_seen_fmri(batch, k_best, device) # in-place
        fmri_prefix = encoder.forward(batch['fmri'].to(device))
        generated_caption = decoder.generate_caption(fmri_prefix, device)
        acc = calculate_semantic_similarity(generated_caption, batch['caption'], device).tolist()
        fields = ['accuracy', 'caption', 'real_caption', 'image', 'fmri']
        d = [dict(zip(fields, t)) for t in zip(acc, generated_caption, batch['caption'], batch['image'], batch['fmri'])] # list of {'accuracy':accuracy, 'generated':generated caption, 'real': labeled caption}
        d.extend(k_best)
        k_best = heapq.nlargest(k, d, key=lambda s:s['accuracy'])

    return k_best

def get_k_best_torch(encoder, decoder, dataloader, k, device):
    k_best = {
        'accuracy': torch.tensor([]).to(device),
        'caption': np.array([]),
        'real_caption': np.array([]),
        'image': torch.tensor([]).to(device),
        'fmri': torch.tensor([]).to(device)
    }
    for batch in dataloader:
        #batch['fmri'] = batch['fmri'].to(device)
        #batch['image'] = batch['image'].to(device)
        remove_duplicates_in_batch(batch) # in-place
        remove_previously_seen_fmri(batch, k_best, device) # in-place
        fmri_prefix = encoder.forward(batch['fmri'].to(device))
        generated_caption = decoder.generate_caption(fmri_prefix, device)
        acc = calculate_semantic_similarity(generated_caption, batch['caption'], device)

        k_best['accuracy'] = torch.cat((k_best['accuracy'], acc), dim=0)
        k_best['caption'] = np.concatenate((k_best['caption'], generated_caption))
        k_best['real_caption'] = np.concatenate((k_best['real_caption'], batch['caption']))
        k_best['image'] = torch.cat((k_best['image'], batch['image'].to(device)), dim=0)
        k_best['fmri'] = torch.cat((k_best['fmri'], batch['fmri'].to(device)), dim=0)

        
        k_to_use = min(k_best['accuracy'].size(dim=0), k) 
        topk_values, topk_indices = torch.topk(k_best['accuracy'], k=k_to_use, dim=0) 


        k_best['accuracy'] = k_best['accuracy'][topk_indices]
        k_best['image'] = k_best['image'][topk_indices]
        k_best['fmri'] = k_best['fmri'][topk_indices]
        k_best['caption'] = k_best['caption'][topk_indices.cpu()]
        k_best['real_caption'] = k_best['real_caption'][topk_indices.cpu()]


    return k_best
        

        

def remove_previously_seen_fmri(batch, k_best, device):
    # Works in-line
    #k_best_fmri = torch.tensor([]) if not k_best else torch.cat([torch.unsqueeze(s['fmri'], dim=0) for s in k_best], dim=0)
    #print(batch['fmri'].device)
    #print(k_best['fmri'].device)
    duplicated_mask = torch.isin(batch['fmri'].to(device), k_best['fmri'])[:, 0, 0]
    duplicated_mask = ~duplicated_mask
    batch['fmri'] = batch['fmri'][duplicated_mask]
    batch['image'] = batch['image'][duplicated_mask]
    batch['caption'] = batch['caption'][duplicated_mask.cpu()]

    return


def remove_duplicates_in_batch(batch):
    # Works in-place
    unique_fmri, idx = unique(batch['fmri'], dim=0)
    batch['fmri'] = unique_fmri
    batch['image']  = torch.index_select(batch['image'], dim=0, index=idx)
    batch['caption'] = np.array(batch['caption'])[idx.cpu()]

    return
    

def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

def objective(encoder, decoder, train_dl, val_dl, device, trial=None):

    # Generate the optimizers
    if trial:
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorial("optimizer", ['Adam', 'AdamW', "SGD"])
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr)
        batch_size = trial.suggest_int("batch_size", 1, 10)
    else:
        lr = LEARNING_RATE
        optimizer = optim.AdamW(encoder.parameters(), lr)
        batch_size = BATCH_SIZE
        #scheduler = get_linear_schedule_with_warmup(
    #optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=NUM_EPOCHS*len(train_dl)
    #)

    decoder.train()
    encoder.eval()
    print("\n\n")
    running_loss = []
    running_semantic_accuracy = []
    val_accuracy = []
    for epoch in range(NUM_EPOCHS):
        print(f"** Starting epoch {epoch} **")
        with tqdm(train_dl, unit='batch') as tepoch:
            semantic_accuracy = 0
            for batch_idx, batch in enumerate(tepoch):
                if batch_idx * batch_size >= TRIAL_NUM_TRAIN_EXAMPLES:
                    break
                
                tepoch.set_description(f"Epoch: {epoch}")

                #batch_fmri = batch['fmri'].to(device)
                batch_fmri = batch['fmri']
                batch_fmri = batch_fmri.to(device)
                batch_caption = batch['caption']

                #print(f">>>> encoding fmri scans ", end="")
                fmri_prefix = encoder.forward(batch_fmri)
                #print(f"-> tokenizing captions ", end="")
                tokens, attention_mask = decoder.tokenizer(batch_caption, return_tensors="pt", padding=True).values()
                tokens, attention_mask, fmri_prefix = tokens.to(device), attention_mask.to(device), fmri_prefix.to(device)
                #print(f"-> decoding ")
                outputs = decoder.forward(tokens, fmri_prefix, attention_mask)
                logits = outputs.logits[:, decoder.prefix_length-1:-1]

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tokens.flatten(),
                    ignore_index=decoder.tokenizer.pad_token_id
                      )
                decoder.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                #scheduler.step()
                #print(f">>>> batch {batch_idx} finished", end="\r")

                # Eval semantic accuracy
                if batch_idx % 10 == 0:
                    with torch.no_grad():
                        generated_caption = decoder.generate_caption(fmri_prefix, device)
                        semantic_accuracy = torch.mean(calculate_semantic_similarity(generated_caption, batch_caption, device)).item()
                        running_semantic_accuracy.append(semantic_accuracy)
                        running_loss.append(loss.item())
                
                tepoch.set_postfix(loss=loss.item(), train_accuracy=semantic_accuracy)

                # Free GPU memory
                del batch_fmri, batch_caption, tokens, attention_mask, logits, outputs, loss
                torch.cuda.empty_cache()
        val_accuracy.append(calculate_accuracy_on_test(encoder, decoder, val_dl, device, return_best_batch=False))
        print(f"---- epoch loss: {np.mean(running_loss[int(np.floor((epoch*len(train_dl))/10 + 1)):-1]):.4}, test accuracy: {np.mean(running_semantic_accuracy[int(np.floor((epoch*len(train_dl))/10 + 1)):-1]):.4}, validation accuracy: {val_accuracy[epoch]:.4} ---- ")

        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_accuracy

def state_dict_MLP_to_MLP_dropout(projection_state_dict, MLP_state_dict):
    new_keys = dict(zip(MLP_state_dict.keys(), projection_state_dict.keys()))
    new_sd = dict((new_keys[key], value) for (key, value) in MLP_state_dict.items())

    return new_sd


#%%
# Imports
import torch
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline

#%%
# List models
checkpoints_path = r"/databases/roeyshafran/BrainCap/Checkpoints/"

checkpoints_files = glob(os.path.join(checkpoints_path, '*.pth'))
len(checkpoints_files)
"""
for checkpoint in checkpoints_files:
    model_dict = torch.load(checkpoint)
    print("-----------------")
    print(f"model {checkpoint}")
    print(model_dict.keys())
    print("-----------------")
    del model_dict
    """

#%% plot training data

batches_in_epoch = 2154//8
#batches_in_epoch = 10000//8

for checkpoint in checkpoints_files[30:-1]:
    model_dict = torch.load(checkpoint)
    training_data = model_dict['training_data']
    model_keys = model_dict.keys()
    projection_sizes = model_dict['decoder_projection']['sizes']
    try:
        comment = model_dict['comment']
    except:
        comment = None
    del model_dict
    torch.cuda.empty_cache()
    running_loss = training_data['running_loss']
    running_semantic_accuracy = training_data['running_semantic_accuracy']
    val_accuracy = training_data['val_accuracy']

    # plot
    fig, axs = plt.subplots(2,1, figsize=(10,10))
    axs = axs.flatten()
    #batch_iterations = np.arange(0, len(train_dl)*NUM_EPOCHS+1, 10)
    batch_iterations = np.arange(0, len(running_loss)*10, 10)
    val_iterations = np.arange(0, len(val_accuracy)*batches_in_epoch, batches_in_epoch)
    axs[0].plot(batch_iterations, running_loss)
    axs[0].set_xlabel('Batch')
    axs[0].set_ylabel('Loss')

    axs[1].plot(batch_iterations, running_semantic_accuracy, label='Train')
    #axs[1].plot(batch_iterations[0:-1:int(np.ceil(len(running_loss)/10))], val_accuracy, '.', label='Validation')
    axs[1].plot(val_iterations, val_accuracy, '-o', label=('Validation', 'Validation - above 0.5'))
    axs[1].set_xlabel('Batch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    fig.suptitle(f"{os.path.basename(checkpoint)}:\nComment: {comment} \nKeys: {model_keys}\n size: {projection_sizes}")
    fig.tight_layout()

#%%
print(checkpoints_files[2])
model_dict = torch.load(checkpoints_files[2])
print(model_dict['scheduler'])
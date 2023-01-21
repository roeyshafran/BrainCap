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

for checkpoint in checkpoints_files:
    model_dict = torch.load(checkpoint)
    print("-----------------")
    print(f"model {checkpoint}")
    print(model_dict.keys())
    print("-----------------")
    del model_dict

#%% plot training data

batches_in_epoch = 10000/8

for checkpoint in checkpoints_files[2:]:
    model_dict = torch.load(checkpoint)
    training_data = model_dict['training_data']
    del model_dict
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
    axs[1].plot(val_iterations, val_accuracy, '.', label='Validation')
    axs[1].set_xlabel('Batch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    fig.suptitle(f"{os.path.basename(checkpoint)}")
    fig.tight_layout()


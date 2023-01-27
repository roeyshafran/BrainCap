# Brain-Cap
Generating Captions for Visual Stimuli Out of fMRI Scans

By Roey Shafran and Yoav Tsoran

This repository is part of a final project in the Technion's course 046211 - Deep Learning

# Overview
Our goal in this project was to develop a model that generates a verbal description of images viewed by subjects, based on their fMRI scans (visual stimulation).
We used the pretrained encoder from the [Mind-Vis](https://github.com/zjc062/mind-vis) article and trained an MLP network in the latent space that transfers the fMRI embedded to GPT2 embedded so that sentences are created at the output of the network.


![Overview](https://user-images.githubusercontent.com/121654746/215019036-d76a7851-c6f3-480c-ac81-8c0b1065c358.png)


Outlines
  1. [Environment setup](#environment-setup)
  2. [Dataset and folder structure](#dataset-and-folder-structure)
  3. [Training](#training)
  4. [Results](#results)
  6. [Credits](#credits)

# Environment setup
 - After cloning into the repository, please run:
    
    ```
    conda env create -f environment.yml
    conda activate brain-cap
    ```
# Dataset and folder structure
Due to size limits, the data and pretrains folders aren't included in this repository ad need to be downloaded seperately.
The data folders include both the fMRI-image datasets used in the MinD-Vis work, a preprocessed dataset we used for training, the captions for the included images and the checkpoints file for our model.
The data folder structure is as follows:
```
data
├── BOLD5000
│   ├── BOLD5000_GLMsingle_ROI_betas
│   ├── BOLD5000_Stimuli
│   ├── COCO-captions
│   │   └── annotations
│   ├── CSI1_no_duplicates.pth
│   └── ImageNet-captions
│       └── imagenet_captions.json
└── Checkpoints
```

The MinD-Vis repositort provides [download links](https://figshare.com/s/94cd778e6afafb00946e) for the fMRI-image datasets and for the pretrained fMRI emcoder. The data.zip file needs to be extracted to this repository data folder as stated above. the fMRI_encoder_pretrain_metafile.pth should be in pretrains folder.
The COCO dataset captions can be downloaded from the [COCO dataset official website](http://images.cocodataset.org/annotations/annotations_trainval2014.zip).
The ImageNet dataset captions can be downloaded from the [mlfoundations/imagenet-captions](https://github.com/mlfoundations/imagenet-captions) GitHub reposiroty.
We also provide a download for our Checkpoints, training dataset (CSI1_no_duplicates.pth) and the MinD-Vis pretrained encoder at ------link--------.

## Training:
- We trained a MLP architecture between the latent spaces while keeping the fMRI encoder and GPT2 decoder freezed.
<p align="center">
  <img src="https://user-images.githubusercontent.com/121654746/215045975-7796091e-0e61-4576-a41b-ac6f0f3a7e5f.png" width="900">
</p>

- To train the model, use the ```train.ipynb``` notebook. This notebook allows training our model on the CSI1_no_duplicates dataset and it is possible to use our last Checkpoints. 
- At the beginning of the notebook, you can change the locations of the different folders if you have downloaded the datasets or checkpoints to a different directory from the repo.
- Example for the training process:
<p align="center">
  <img src="https://user-images.githubusercontent.com/121654746/215046105-34fb4a45-aa9e-43bd-84d5-eda8c37639de.png" width="500">
</p>


## Results
<p align="center">
  <img src="https://user-images.githubusercontent.com/121654746/215047584-731b1364-7aa1-48bc-ae7e-f3b881aa72ca.png" width="500">
</p>


## Credits:
Other than repositories previously acknowledged:
 - The model being trained takes inspiration from a model presented in ["ClipCap: CLIP Prefix for Image Captioning"](https://github.com/rmokady/CLIP_prefix_caption)

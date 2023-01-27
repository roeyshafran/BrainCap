# Brain-Cap
Generating Captions for Visual Stimuli Out of fMRI Scans

By Roey Shafran and Yoav Tsoran

This repository is part of a final project in the Technion's course 046211 - Deep Learning

# Overview
This project presents a proposed method for creating a descriptive text of the visual stimuli presented to a subject during an fMRI scan. The method is based on the combination of two previous works, MinD-Vis and ClipCap. [Mind-Vis](https://github.com/zjc062/mind-vis) is used to generate meaningful embeddings for fMRI scans, while [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) is used to create an image embedding that is used as a prefix to GPT2 pre-trained language model. Our work builds upon these previous methods by using the ClipCap method in an fMRI-to-caption setup and training a simple mapping network between the MinD-Vis fMRI encoder embedding space and the GPT2 embedding space. This approach can help improve our understanding of the brain's visual system and explore potential technological applications.



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
 - As an alternative, using pip you can run:
    ```
    pip install -r requirements.yml
    ```
# Dataset and folder structure
Due to size limits, the data and pretrains folders aren't included in this repository ad need to be downloaded seperately.
The data folder include both the fMRI-image datasets used in the MinD-Vis work, the captions for the included images and the checkpoints file for our model.
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

The MinD-Vis repository provides [download links](https://figshare.com/s/94cd778e6afafb00946e) for the fMRI-image datasets. The data.zip file needs to be extracted to this repository data folder as stated above. 
The COCO dataset captions can be downloaded from the [COCO dataset official website](http://images.cocodataset.org/annotations/annotations_trainval2014.zip).
The ImageNet dataset captions can be downloaded from the [mlfoundations/imagenet-captions](https://github.com/mlfoundations/imagenet-captions) GitHub reposiroty.
We also provide a download for our [Checkpoints](https://figshare.com/articles/software/BrainCap_checkpoints_and_pretrains/21966860), and the MinD-Vis [pretrained encoder](https://figshare.com/articles/software/BrainCap_-_pretrained_fMRI_encoder_from_MinD-Vis/21967490).
The fMRI_encoder_pretrain_metafile.pth should be copied to the pretrains folder.

## Creating the Dataset
- To speed up the training we use only one caption for each training sample and save the preprocessed dataset for faster loading.
- A script for creating the dataset file is provided.
- For example, to create the dataset file only for the first subject (CSI1), as used for our training, please run the following line from the code folder:
  ```
  python create_dataset_no_dup.py --path ../data/BOLD5000 --save-path ../data/BOLD5000/CSI1_no_duplicates.pth --subjects CSI1 --batch_size 8
  ```
- If you saved the BOLD5000 folder at a different location, want to train the model on more subjects or use larger batch size for the preprocess (might help with the script running speed) run the script with the --help flag.

## Training:
- We trained a MLP architecture between the latent spaces while keeping the fMRI encoder and GPT2 decoder freezed.
<p align="center">
  <img src="https://user-images.githubusercontent.com/121654746/215045975-7796091e-0e61-4576-a41b-ac6f0f3a7e5f.png" width="900">
</p>

- To train the model, use the ```train.ipynb``` notebook. This notebook allows training our model on the CSI1_no_duplicates dataset from scratch orkeep training from our last Checkpoints. 
- At the beginning of the notebook, you can change the locations of the different folders if you have downloaded the datasets or checkpoints to a different directory from the repo.
- Example for the training process:
<p align="center">
  <img src="https://user-images.githubusercontent.com/121654746/215046105-34fb4a45-aa9e-43bd-84d5-eda8c37639de.png" width="500">
</p>


## Results
<p align="center">
  <img src="https://user-images.githubusercontent.com/121654746/215047584-731b1364-7aa1-48bc-ae7e-f3b881aa72ca.png" width="500">
</p>


## References:
 -  Chen, Z., Qing, J., Xiang, T., Yue, W., & Zhou, J. (2022). Seeing Beyond the Brain: Masked Modeling Conditioned Diffusion Model for Human Vision Decoding. In arXiv.
 - Mokady, R., Hertz, A., & Bermano, A. (2021). Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734.

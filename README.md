# Novel Lobe-based Transformer Model (LobTe) to Predict Emphysema Progression
This repository implements a Lobe Transformer Encoder (ViT) to predict the change in lung density at five years from a baseline chest CT scans.

Within this repository, we train two distinct LobTe workflows: one designed to predict changes across the entire lung (LobTe_Lung/), and another fine-tuned for individual lung lobes (LobTe_Lobes/). Both approaches heavily utilize and share the same local foundational model, which captures the baseline evolution of lung density.

### Dataset
To train the models we recommend to use the CT scans from phase 1 and 2 from
the COPDGene study. Follow the instruction in www.copdgene.org to get access to
the images.

### Requirements
Make sure you have the following libraries installed:
* **Tensorflow** 2.12.1
* **Pytorch** 3.13
* **Numpy** 1.23.2
* **SimpleITK** 2.3.0
* **scikit-learn** 1.1.2

## The Shared Local Foundational Model
The core of both LobTe prediction pipelines relies on a shared local foundational model developed for general-purpose representation learning in smokers, both with and without COPD. Built on an autoencoder framework, this model is trained end-to-end in an unsupervised manner using a random subset of co-registered local CT patches (32×32 pixels, 0.64 mm²).

Rather than being explicitly trained to predict local emphysema progression from the start, it focuses on general feature extraction by reconstructing the local neighborhood at the 5-year follow-up. By learning these structural evolutions, the model serves as a robust, general-purpose foundation for downstream density prediction tasks.

Once the local foundational model is initially trained (LobTe_Lung/train_AE.py), it is then fine-tuned to capture local emphysema progression and define the final local density model using the LobTe_Lung/train_AER.py script.
![LobTe workflow](/assets/images/LFM_workflow.png)


## Whole-Lung Prediction Workflow
Designed to predict density changes across the entire lung, this workflow leverages the shared local foundational model to capture the underlying evolution of lung density and emphysema progression. A transformer-based architecture then processes these insights to predict the overall change in lung density (based on the 15th Percentile criteria, ΔALD), driven by the specific extent of tissue destruction within each lobe.

![LobTe workflow](/assets/images/LobTe_workflow.png)

## Train
1. Use ythe script LobeTe_Lung/create_fingerprint_by_lobe.py for creating the lobe fingerprints.
2. Train the LobTe model using the script LobeTe_Lung/train_LobTe.py

## Inference
1. For a specific chest CT scan, use the LobeTe_Lung/create_fingerprint_by_lobe.py script to generate the lobe fingerprints.
2. Predict the change in adjusted lung density at five years using the LobeTe_Lung/lobTe_prediction.py script.

## Lobe-Specific Prediction Workflow
This model leverages a global self-attention mechanism to predict annualized, lobe-specific changes in volume-adjusted lung density (ΔALD). It achieves this by processing the five lobe fingerprints through a transformer-based architecture to capture density changes  within each individual lobe.
![LobTe workflow](/assets/images/LobTe_workflow_lobes.png)

## Train
1. Train the LobTe model using the script LobeTe_Lobes/train_LobTe.py

## Inference
1. For a specific chest CT scan, use the LobeTe_Lung/create_fingerprint_by_lobe.py script to generate the lobe fingerprints.
2. Predict the 5-year change in adjusted lung density for each lobe using the LobeTe_Lung/lobTe_prediction.py script.

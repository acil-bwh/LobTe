# Novel Lobe-based Transformer Model (LobTe) to predict Emphysema progression
This reposiroy implement a lobe transformer encoder (ViT) to poredict the
change of lung density at five years from a CT scan. Figure 1 show the workflow
proposed: (a) A local density model is utilized to capture the evolution of lung
density and the progression of emphysema. (b) A transformer-based model is designed to predict changes in lung density (∆ALD) based on the extent of tissue destruction within each lung lobe.

![LobTe workflow](/assets/images/LobTe_workflow.png)

## Requirements
Tensorflow 2.12.1

Numpy 1.23.2

SimpleITK 2.3.0

scikit-learn 1.1.2

## Dataset
To train the models we recommend to use the CT scans from phase 1 and 2 from
the COPDGene study. Follow the instruction in www.copdgene.org to get access to
the images.

## Training
1. Pre-train the local longirudinal autoencoder using the script
   train_AE.py
2. Train the local density model using the script train_AER.py
3. Create the lobe embedding fingerprint for each lobes using the script
   create_fingerprint_by_lobe.py
4. Train the LobTe model using the script train_LobTe.py

## Inference
1. Create the lobe fingerorint for a particular subject using the script
   create_fingerprint_by_lobe.py
2. Predict the change of adjust lung density at five years using the script
   lobTe_prediction.py

# Code for SA-MGRU: Gate-Aligned Fusion of Self-Attention and Multi-Gate GRU for Medical Image Segmentation

## Dataset 
In the case of input, we evaluate the performance of our proposed modules on the following datasets:  
- **The synapse dataset can be found at the [repo of TransUNet](https://github.com/Beckschen/TransUNet).**  
- **ISIC2018 dataset: This dataset focuses on the segmentation of skin lesions [link](https://challenge.isic-archive.com/landing/2018/).**   
- **CVC-ClinicDB dataset: This dataset pertains to polyp segmentation in colonoscopy videos and has been used for comparing automatic segmentation methods. [link](https://polyp.grand-challenge.org/CVCClinicDB/).**  
- **The ACDC dataset can be found at the [repo of MTUNet](https://github.com/Dootmaan/MT-UNet).**  

## Usage
Recommended environment:

```text
Python 3.8
Pytorch 1.11.0
torchvision 0.12.0
```

Please use `pip install -r requirements.txt` to install the dependencies.

Data preparation:

- Synapse Multi-organ dataset: Please place the dataset into the `Synapse/data/` directory.


- ACDC dataset: Please place the dataset into the `ACDC/data/` directory.

Pretrained model:

Please download the pretrained MaxViT models from [Google Drive](https://drive.google.com/drive/folders/1LG2hl1fx17_oLQKxRMZeido-TehHFoiF?usp=drive_link), and then put it in the `Synapse/pretrained_pth/maxvit/` folder for initialization.

Training:

- To train on the ACDC dataset, run from the project root:

```powershell
python ./code/ACDC/train.py
```

- To train on the Synapse dataset, run from the project root:

```powershell
python ./code/Synapse/train_synapse.py
```

Evaluation:

You can download our trained weights for ACDC and Synapse experiments from [this Google Drive link](https://drive.google.com/drive/folders/1LG2hl1fx17_oLQKxRMZeido-TehHFoiF?usp=drive_link).
Alternatively, you can use your own trained weights and run the corresponding test script to evaluate the model.
```powershell
python ./code/ACDC/test_model.py
```

```powershell
python ./code/Synapse/test_synapse.py
```

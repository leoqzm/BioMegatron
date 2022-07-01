# BioMegatron
Rebuild biomegatron tutorial
## 1. Token Classification 
In the ```Token_Classification_BioMegatron.ipynb ``` notebook, it shows an example for how BioMegatron model train and inference a named-entity recoganization task.
###environment set up
For running the notebook by ourselves in google colab, we first need to use 
```
!nvcc --version
import torch
print(torch.__version__)

```
to check pytorch binary file is compiled under the right cuda version, or apex will build failed because of this problem. I used 
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:15:13_PDT_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0
torch version = 1.11.0+cu113
```
Then we need to run the code block
```
!git clone https://github.com/NVIDIA/apex
!cd apex
!git checkout 5d8c8a8eedaf567d56f0762a45431baf9c0e800e
!pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" /content/apex/
```
to install the apex package which is required for training the biomegatron model. <mark> Make sure install the apex package before the import block(show below)</mark>

```
from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager

import os
import wget
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
```

Then the notebook will run correctly.
###
### Directory Set Up
For convinient to save checkpoints or results, I linked the colab with google drive, and changed directory path with the google drive directory.

To mount with google drive:
```
from google.colab import drive
drive.mount('/content/drive')
```

And edit work directory and data directory path, I change these two line of code:
```
DATA_DIR = os.path.join(os.getcwd(), 'DATA_DIR')

WORK_DIR = os.path.join(os.getcwd(), 'WORK_DIR')
```
TO
```
DATA_DIR = os.path.join(MAIN_DIR, 'DATA_DIR')
RE_DATA_DIR = os.path.join(DATA_DIR, 'RE')
WORK_DIR = os.path.join(MAIN_DIR, 'WORK_DIR')
```
In your process, you can change your ```MAIN_DIR``` to your preferred directory with
```
MAIN_DIR = 'YOUR_PATH_TO_DIRECTORY'
```
###
### Input Output Example
The task is a simplified task from the simplified NCBI disease dataset. Which the input was supposed to be
``` HTML
<category = "Modifier">adenomatous polyposis coli tumour</category>
<category = "Modifier">adenomatous polyposis coli ( APC ) tumour</category>
<category = "Modifier">colon carcinoma</category>
<category = "Modifier">colon carcinoma</category>
<category = "SpecificDisease">cancer</category>
```
But for purposes, they simplified the  identified category (such as "Modifier", "Specific Disease", and a few others) to generally be a "disease".

Also the input become in a IOB tagging way
```text
Identification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor .
O              O  O    O O O         O  O   B           I         I    I      O          O  
```

The output also become in a same tagging way with IOB tagging.
###
##


## 2. Relation Extraction
In the ```Relation_Extraction_BioMegatron.ipynb ``` notebook, it shows an example for how BioMegatron model train and inference a named-entity recoganization task.
### environment set up
For running the notebook by ourselves in google colab, we first need to use 
```
!nvcc --version
import torch
print(torch.__version__)

```
to check pytorch binary file is compiled under the right cuda version, or apex will build failed because of this problem. I used 
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:15:13_PDT_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0
torch version = 1.11.0+cu113
```
Then we need to run the code block
```
!git clone https://github.com/NVIDIA/apex
!cd apex
!git checkout 5d8c8a8eedaf567d56f0762a45431baf9c0e800e
!pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" /content/apex/
```
to install the apex package which is required for training the biomegatron model. <mark> Make sure install the apex package before the import block(show below)</mark>

```
from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager

import os
import wget
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
```

Then the notebook will run correctly.
###
### Directory Set Up
For convinient to save checkpoints or results, I linked the colab with google drive, and changed directory path with the google drive directory.

To mount with google drive:
```
from google.colab import drive
drive.mount('/content/drive')
```

And edit work directory and data directory path, I change these two line of code:
```
DATA_DIR = os.path.join(os.getcwd(), 'DATA_DIR')

WORK_DIR = os.path.join(os.getcwd(), 'WORK_DIR')
```
TO
```
DATA_DIR = os.path.join(MAIN_DIR, 'DATA_DIR')
RE_DATA_DIR = os.path.join(DATA_DIR, 'RE')
WORK_DIR = os.path.join(MAIN_DIR, 'WORK_DIR')
```
In your process, you can change your ```MAIN_DIR``` to your preferred directory with
```
MAIN_DIR = 'YOUR_PATH_TO_DIRECTORY'
```
###

### Epoch edited
After the environment and directory built, the training epoch number in model config also need to be edited.
I add a line in Model Configuration block

```
config.trainer.max_epochs=3
```
which set max training epoch to be 3, the original 100 epoch is needless because too much epoch will lead overfitting
###

### Input Output Example
The task is to classify the relation of a [GENE] and [CHEMICAL] in a sentence from [BioCreative VI website](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/), for example like the following:
```html
14967461.T1.T22	<@CHEMICAL$> inhibitors currently under investigation include the small molecules <@GENE$> (Iressa, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as cetuximab (IMC-225, Erbitux).	<CPR:4>
14967461.T2.T22	<@CHEMICAL$> inhibitors currently under investigation include the small molecules gefitinib (<@GENE$>, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as cetuximab (IMC-225, Erbitux).	<CPR:4>
```
AND the relation class have  following 5 classes:


| Relation Class      | Relations |
| ----------- | ----------- |
| CPR:3      |  Upregulator and activator       |
| CPR:4   | Downregulator and inhibitor         |
| CPR:5 | Agonist |
| CPR:6 | Antagonist |
| CPR:9 | Substrate and product of |

But we used a simplified ChemProt dataset version, and the training input example is :
```
BC6ENTG antagonists have been utilized , following their initial chemical synthesis in 1933 , both in the treatment of conditions in which BC6ENTC is considered to be of pathogenic importance and conversely to help elucidate the role of BC6OTHER in disease , through an evaluation of their influence on disease expression.	0
```
Which is only a string follow by a label number
And the label mapping changed to the following format:
```
CPR:5	0
False	1
CPR:6	2
CPR:9	3
CPR:4	4
CPR:3	5
```

###
### Evaluation Set up
In the original notebook, it does not have the test evalution part for the model. 
I add the evaluation part with these line of codes:

```
config.model.test_ds.file_path = os.path.join(RE_DATA_DIR, 'test.tsv')
model.setup_test_data(test_data_config=config.model.test_ds)

trainer.test(model=model, ckpt_path=None)
```

and the evaluation output is in this style:

```
[NeMo I 2022-06-30 23:57:17 text_classification_model:144] test_report: 
    label                                                precision    recall       f1           support   
    label_id: 0                                             70.77      75.41      73.02        183
    label_id: 1                                             93.68      92.00      92.83      10956
    label_id: 2                                             77.81      85.27      81.37        292
    label_id: 3                                             64.09      63.20      63.64        644
    label_id: 4                                             76.97      83.84      80.25       1658
    label_id: 5                                             72.29      74.25      73.25        664
    -------------------
    micro avg                                               88.60      88.60      88.60      14397
    macro avg                                               75.93      78.99      77.39      14397
    weighted avg                                            88.83      88.60      88.69      14397
    
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
         test_f1             88.60179138183594
        test_loss           0.3601756989955902
     test_precision          88.60179138183594
       test_recall           88.60179138183594
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
[{'test_f1': 88.60179138183594,
  'test_loss': 0.3601756989955902,
  'test_precision': 88.60179138183594,
  'test_recall': 88.60179138183594}]
  ```
  ###
  ##

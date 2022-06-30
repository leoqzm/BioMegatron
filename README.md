# BioMegatron
Rebuild biomegatron tutorial
## Token Classification 
In the ```Token_Classification_BioMegatron.ipynb ``` notebook, it shows an example for how BioMegatron model train and inference a named-entity recoganization task.

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



## Relation Extraction
In the ```Relation_Extraction_BioMegatron.ipynb ``` notebook, it shows an example for how BioMegatron model train and inference a named-entity recoganization task.

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

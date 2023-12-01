# CS 490 - RnD Project I (Autumn 2023)

This project aims to diagnose the domain robustness of LXMERT ["LXMERT: Learning Cross-Modality Encoder Representations from Transformers"](https://arxiv.org/abs/1908.07490)

Pre-training weights can be found at: [`https://nlp.cs.unc.edu/data/github_pretrain/lxmert20/Epoch20_LXRT.pth`](http://nlp.cs.unc.edu/data/model_LXRT.pth).

## Fine-tune on Vision-and-Language Tasks
The fine-tuning of the LXMERT pre-trained model is done with the following hyper-parameters:

|Dataset      | Batch Size   | Learning Rate   | Epochs  | Load Answers  |
|---   |:---:|:---:   |:---:|:---:|
|VQA   | 32  | 5e-5   | 4   | Yes |

### General 
**Following are the directions to fine-tune the model, as given by the authors.**

The code requires **Python 3** and please install the Python dependencies with the command:
```bash
pip install -r requirements.txt
```

By the way, a Python 3 virtual environment could be set up and run with:
```bash
virtualenv name_of_environment -p python3
source name_of_environment/bin/activate
```
### VQA
#### Fine-tuning
1. Please make sure the LXMERT pre-trained model is either [downloaded](#pre-trained-models) or [pre-trained](#pre-training).

2. Download the re-distributed json files for VQA 2.0 dataset. The raw VQA 2.0 dataset could be downloaded from the [official website](https://visualqa.org/download.html).
    ```bash
    mkdir -p data/vqa
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/
    ```
3. Download faster-rcnn features for MS COCO train2014 (17 GB) and val2014 (8 GB) images (VQA 2.0 is collected on MS COCO dataset).
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
    ```bash
    mkdir -p data/mscoco_imgfeat
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
    ```

4. Before fine-tuning on whole VQA 2.0 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `vqa_lxr955_tiny` is the name of this experiment.
    ```bash
    bash run/vqa_finetune.bash 0 vqa_lxr955_tiny --tiny
    ```
5. If no bug came out, then the model is ready to be trained on the whole VQA corpus:
    ```bash
    bash run/vqa_finetune.bash 0 vqa_lxr955
    ```
It takes around 8 hours (2 hours per epoch * 4 epochs) to converge. 
The **logs** and **model snapshots** will be saved under folder `snap/vqa/vqa_lxr955`. 
The validation result after training will be around **69.7%** to **70.2%**. 

#### Local Validation
The results on the validation set (our minival set) are printed while training.
The validation result is also saved to `snap/vqa/[experiment-name]/log.log`.
If the log file was accidentally deleted, the validation result in training is also reproducible from the model snapshot:
```bash
bash run/vqa_test.bash 0 vqa_lxr955_results --test minival --load snap/vqa/vqa_lxr955/BEST
```

### General Debugging Options
Since it takes a few minutes to load the features, the code has an option to prototype with a small amount of
training data. 
```bash
# Training with 512 images:
bash run/vqa_finetune.bash 0 --tiny 
# Training with 4096 images:
bash run/vqa_finetune.bash 0 --fast
```

## Alternative Dataset and Features Download Links 
All default download links are provided by our servers in [UNC CS department](https://cs.unc.edu) and under 
our [NLP group website](https://nlp.cs.unc.edu) but the network bandwidth might be limited. 
We thus provide a few other options with Google Drive and Baidu Drive.

The files in online drives are almost structured in the same way 
as our repo but have a few differences due to specific policies.
After downloading the data and features from the drives, 
please re-organize them under `data/` folder according to the following example:
```
REPO ROOT
 |
 |-- data                  
 |    |-- vqa
 |    |    |-- train.json
 |    |    |-- minival.json
 |    |    |-- nominival.json
 |    |    |-- test.json
 |    |
 |    |-- mscoco_imgfeat
 |    |    |-- train2014_obj36.tsv
 |    |    |-- val2014_obj36.tsv
 |    |    |-- test2015_obj36.tsv
 |    |
 |    |-- vg_gqa_imgfeat -- *.tsv
 |    |-- gqa -- *.json
 |    |-- nlvr2_imgfeat -- *.tsv
 |    |-- nlvr2 -- *.json
 |    |-- lxmert -- *.json          # Pre-training data
 | 
 |-- snap
 |-- src
```

Please also kindly contact us if anything is missing!

### Google Drive
As an alternative way to download feature from our UNC server,
you could also download the feature from google drive with link [https://drive.google.com/drive/folders/1Gq1uLUk6NdD0CcJOptXjxE6ssY5XAuat?usp=sharing](https://drive.google.com/drive/folders/1Gq1uLUk6NdD0CcJOptXjxE6ssY5XAuat?usp=sharing).
The structure of the folders on drive is:
```
Google Drive Root
 |-- data                  # The raw data and image features without compression
 |    |-- vqa
 |    |-- gqa
 |    |-- mscoco_imgfeat
 |    |-- ......
 |
 |-- image_feature_zips    # The image-feature zip files (Around 45% compressed)
 |    |-- mscoco_imgfeat.zip
 |    |-- nlvr2_imgfeat.zip
 |    |-- vg_gqa_imgfeat.zip
 |
 |-- snap -- pretrained -- model_LXRT.pth # The pytorch pre-trained model weights.
```
Note: image features in zip files (e.g., `mscoco_mgfeat.zip`) are the same to which in `data/` (i.e., `data/mscoco_imgfeat`). 
If you want to save network bandwidth, please download the feature zips and skip downloading the `*_imgfeat` folders under `data/`.
### Baidu Drive

Since [Google Drive](
https://drive.google.com/drive/folders/1Gq1uLUk6NdD0CcJOptXjxE6ssY5XAuat?usp=sharing
) is not officially available across the world,
we also create a mirror on Baidu drive (i.e., Baidu PAN). 
The dataset and features could be downloaded with shared link 
[https://pan.baidu.com/s/1m0mUVsq30rO6F1slxPZNHA](https://pan.baidu.com/s/1m0mUVsq30rO6F1slxPZNHA) 
and access code `wwma`.
```
Baidu Drive Root
 |
 |-- vqa
 |    |-- train.json
 |    |-- minival.json
 |    |-- nominival.json
 |    |-- test.json
 |
 |-- mscoco_imgfeat
 |    |-- train2014_obj36.zip
 |    |-- val2014_obj36.zip
 |    |-- test2015_obj36.zip
 |
 |-- vg_gqa_imgfeat -- *.zip.*  # Please read README.txt under this folder
 |-- gqa -- *.json
 |-- nlvr2_imgfeat -- *.zip.*   # Please read README.txt under this folder
 |-- nlvr2 -- *.json
 |-- lxmert -- *.json
 | 
 |-- pretrained -- model_LXRT.pth
```

Since Baidu Drive does not support extremely large files, 
we `split` a few features zips into multiple small files. 
Please follow the `README.txt` under `baidu_drive/vg_gqa_imgfeat` and 
`baidu_drive/nlvr2_imgfeat` to concatenate back to the feature zips with command `cat`.

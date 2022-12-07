# CS470_K_SAM

Hello everyone. We are team **Oldboys**, and it is our project results **K-SAM**.

It is Face Aging Model with AI_HUB's Korean face datasets.

All information about K-SAM is described below.

![Jang1](https://user-images.githubusercontent.com/71695489/205508296-aa60af72-7182-45d8-af4d-a8ef001bb83e.jpg)

<br/>

Here below link for the poster shows the detailed description. <br/>
=> [Style-based Age Manipulation Poster](https://docs.google.com/drawings/d/1nRQFLXTqVGUcLOyQGcd1hXAuTmZ-aQles3_a-YJdyZc/edit)

<br/>

Base paper for this Face Aging model is written and linked below. <br/>
 => [Only a Matter of Style: Age Transformation Using a Style-Based Regression Model](https://arxiv.org/pdf/2102.02754.pdf)
 
<br/>

The model configure and architecture is following the official github of the paper. <br/>
**Read below for our addition first, and follows official guide below.** <br/>
 => [official link of SAM - "yuval-alaluf / SAM"](https://github.com/yuval-alaluf/SAM)
 
<br/>

You can download the pre-trained model used from here. <br/>
 => [Get pre-trained model here](https://github.com/yuval-alaluf/SAM#pretrained-models)
 
<br/>

In our trained model, we used 5098 images for training and 1068 images for validation. <br/>
You also can download the pre-trained K-SAM model used for CS470 final project here. <br/>
 => [Get pre-trained K-SAM model here](https://drive.google.com/file/d/1v_ABip_aG9ZD3IMxYH4qSBQI0PfuKGQw/view?usp=share_link)

<br/>
 
You can compare the whole generated images by assembling with **"/results/assemble.py"**  
You should **modify the code to fit your own datasets**.

<br/>

Used train, test(=validation), result images with link listed below.<br/>
 => [train datasets](https://drive.google.com/file/d/1OtCT7v3OpiC-92A-Bdb2_HEbOKn5nTiP/view?usp=share_link)<br/>
 => [test(=validation) datasets](https://drive.google.com/file/d/1eco4WBUu1VZ1a5_-gHLwIASMIrbw3NlE/view?usp=share_link)<br/>
 => [**result face aging images**](https://drive.google.com/file/d/1G_7Wz8AgOJDBCsiuOwByu4xyAuwdrmK5/view?usp=share_link)<br/>

## Read carefully below to fully understand what we added.

### 1. Image Crop
##### Image Crop for Training

- You can explicitly crop your personal "Face Train Images" using **"/images/cropper.py"**
- You should **modify the code to fit your own datasets**.
- **"/face_detection.xml"** was used for CascadeClassifier in face detection. 
- This file was based on [**OpenCV officail github**](https://github.com/opencv/opencv/tree/master/data/haarcascades).<br/>


##### Image Crop for Inference

- In **/datasets/inference_dataset.py**, you can annotate or not to choose whether activate cropping or not for the inference images.


### 2. Korean Image Dataset

- You can download the original dataset used in this project from here.
  [AI hub](https://aihub.or.kr/)
- Join there, find and download **가족관계가 알려진 얼굴 이미지 데이터**.

![image](https://user-images.githubusercontent.com/71695489/205508518-011a5f5d-3d63-4fb6-8bde-1dea8a0106cd.png)

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- Python 3

### Installation
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `/environment/ksam_env.yml`.

## Pretrained Models
Please download the pretrained aging model from the following links.

| Path | Description
| :--- | :----------
|[SAM](https://drive.google.com/file/d/1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC/view?usp=sharing)  | SAM trained on the FFHQ dataset for age transformation.

You can run this to download it to the right place:

```
mkdir pretrained_models
pip install gdown
gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt
wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
```

In addition, we provide various auxiliary models needed for training your own SAM model from scratch.  
This includes the pretrained pSp encoder model for generating the encodings of the input image and the aging classifier 
used to compute the aging loss during training.

| Path | Description
| :--- | :----------
|[pSp Encoder](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing) | pSp taken from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) trained on the FFHQ dataset for StyleGAN inversion.
|[FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during training.
|[VGG Age Classifier](https://drive.google.com/file/d/1atzjZm_dJrCmFWCqWlyspSpr3nI6Evsh/view?usp=sharing) | VGG age classifier from DEX and fine-tuned on the FFHQ-Aging dataset for use in our aging loss

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. 
However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 

## Training
### Preparing your Data
Please refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and inference.   
Then, refer to `configs/data_configs.py` to define the source/target data paths for the train and test sets as well as the 
transforms to be used for training and inference.
    
We first went to `configs/paths_config.py` and define:
``` 
dataset_paths = {
    'celeba_test': 'images/new_valid_10',
    'ffhq': 'images/new_train',
}
```
Then, in `configs/data_configs.py`, we define:
```
DATASETS = {
	'ffhq_aging': {
		'transforms': transforms_config.AgingTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	}
}
``` 
When defining the datasets for training and inference, we will use the values defined in the above dictionary.

### Image Crop
Please refer to `images/cropper.py` to set the appropriate path to your training image dataset.   
Our training images are in the example form of `images/valid/0021to0040/TS0028/A/3.Age/F0028_AGE_M_59_f1.jpg`.   
You should check the directory of both training and validation images on the variable `img_folder_path`.

### Training SAM
The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.

Training SAM with the settings used in the paper can be done by running the following command:
```
python scripts/train.py \
--dataset_type=ffhq_aging \
--exp_dir=results/new_train \
--workers=6 \
--batch_size=6 \
--test_batch_size=6 \
--test_workers=6 \
--val_interval=500 \
--save_interval=1500 \
--start_from_encoded_w_plus \
--id_lambda=0.1 \
--lpips_lambda=0.1 \
--lpips_lambda_aging=0.1 \
--lpips_lambda_crop=0.6 \
--l2_lambda=0.25 \
--l2_lambda_aging=0.25 \
--l2_lambda_crop=1 \
--w_norm_lambda=0.005 \
--aging_lambda=5 \
--cycle_lambda=1 \
--input_nc=4 \
--target_age=uniform_random \
--use_weighted_id_loss
```

## Testing
### Inference
Having trained your model or if you're using a pretrained SAM model, you can use `scripts/inference.py` to run inference
on a set of images.   
We used the best model which is the result of 18000th iteration.   
We used images for testing in `images/members`.

```
python scripts/inference.py \
--exp_dir=results/k_crop_member \
--checkpoint_path=results/new_train/checkpoints/iteration_18000.pt \
--data_path=images/members \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs
--target_age=0,10,20,30,40,50,60,70,80
```

You can compare these results with the original pretrained checkpoint in `/root/SAM/pretrained_models/sam_ffhq_aging.pt`.

## Credits
**StyleGAN2 model and implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon  
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**IR-SE50 model and implementations:**  
https://github.com/TreB1eN/InsightFace_Pytorch  
Copyright (c) 2018 TreB1eN  
License (MIT) https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE  

**Ranger optimizer implementation:**  
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer   
License (Apache License 2.0) https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/LICENSE  

**LPIPS model and implementation:**  
https://github.com/S-aiueo32/lpips-pytorch  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE  

**DEX VGG model and implementation:**  
https://github.com/InterDigitalInc/HRFAE  
Copyright (c) 2020, InterDigital R&D France  
https://github.com/InterDigitalInc/HRFAE/blob/master/LICENSE.txt

**pSp model and implementation:**   
https://github.com/eladrich/pixel2style2pixel  
Copyright (c) 2020 Elad Richardson, Yuval Alaluf  
https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE

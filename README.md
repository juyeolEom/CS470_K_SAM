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

### 1. Environment
- Use conda environment and install required models defined in **/environment/ksam_env.yml**.

### 2. Image Crop
##### Image Crop for Training

- You can explicitly crop your personal "Face Train Images" using **"/images/cropper.py"**
- You should **modify the code to fit your own datasets**.
- **"/face_detection.xml"** was used for CascadeClassifier in face detection. 
- This file was based on [**OpenCV officail github**](https://github.com/opencv/opencv/tree/master/data/haarcascades).<br/>


##### Image Crop for Inference

- In **/datasets/inference_dataset.py**, you can annotate or not to choose whether activate cropping or not for the inference images.


### 3. Korean Image Dataset

- You can download the original dataset used in this project from here.
  [AI hub](https://aihub.or.kr/)
- Join there, find and download **가족관계가 알려진 얼굴 이미지 데이터**.

![image](https://user-images.githubusercontent.com/71695489/205508518-011a5f5d-3d63-4fb6-8bde-1dea8a0106cd.png)

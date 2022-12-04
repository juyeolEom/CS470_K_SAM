import os
import cv2
import numpy
from matplotlib import pyplot as plt

img_name = 'Lee1.png'

root = 'root/SAM/results'
pre_non_valid = '/root/SAM/results/pre_non_member'
pre_crop_valid = '/root/SAM/results/pre_crop_member'
k_crop_valid = '/root/SAM/results/k_crop_member'

pre_non_img = []
pre_crop_img = []
k_crop_img = []

for n in ['10', '20', '30', '40', '50', '60', '70', '80']:
  if n == '10':
    prev_img_1 = cv2.imread(os.path.join(pre_non_valid, 'inference_coupled', n, img_name))
    print(os.path.join(pre_non_valid, 'inference_results', n, img_name))
  else:
    new_img_1 = cv2.imread(os.path.join(pre_non_valid, 'inference_results', n, img_name))
    prev_img_1 = cv2.hconcat([prev_img_1, new_img_1])
prev_img_1 = cv2.resize(prev_img_1, (1350, 150))

for n in ['10', '20', '30', '40', '50', '60', '70', '80']:
  if n == '10':
    prev_img_2 = cv2.imread(os.path.join(pre_crop_valid, 'inference_coupled', n, img_name))
    print(os.path.join(pre_crop_valid, 'inference_results', n, img_name))
  else:
    new_img_2 = cv2.imread(os.path.join(pre_crop_valid, 'inference_results', n, img_name))
    prev_img_2 = cv2.hconcat([prev_img_2, new_img_2])
prev_img_2 = cv2.resize(prev_img_2, (1350, 150))

for n in ['10', '20', '30', '40', '50', '60', '70', '80']:
  if n == '10':
    prev_img_3 = cv2.imread(os.path.join(k_crop_valid, 'inference_coupled', n, img_name))
    print(os.path.join(k_crop_valid, 'inference_results', n, img_name))
  else:
    new_img_3 = cv2.imread(os.path.join(k_crop_valid, 'inference_results', n, img_name))
    prev_img_3 = cv2.hconcat([prev_img_3, new_img_3])
prev_img_3 = cv2.resize(prev_img_3, (1350, 150))

prev_img = cv2.vconcat([prev_img_1, prev_img_2, prev_img_3])

#cv2.imshow('inference_results', prev_img)
cv2.imwrite('/root/SAM/results/assemble/'+img_name, prev_img)
#cv2.waitKey(0)

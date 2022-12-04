import os
import numpy as np
from PIL import Image
import cv2
import sys
from tqdm import tqdm

pbar = tqdm(total = 5000)

img_folder_path = './images/valid'
cropped_img_path = './images/new_valid'

### Code for cropping train images ###
## To crop your individual image datasets, make your own crop code with below ##
################################################################################
'''
### Code for 
for nton in os.listdir(img_folder_path):
    for TSn in os.listdir(os.path.join(img_folder_path, nton)):
        for AB in os.listdir(os.path.join(img_folder_path, nton, TSn)):
            if os.path.isdir(os.path.join(img_folder_path, nton, TSn, AB, '3.Age')):
                for imgs in os.listdir(os.path.join(img_folder_path, nton, TSn, AB, '3.Age')):
                    if os.path.isfile(os.path.join(img_folder_path, nton, TSn, AB, '3.Age', imgs)):
                        from_im = Image.open(os.path.join(img_folder_path, nton, TSn, AB, '3.Age', imgs))
                        w, h = from_im.size
                        np_image = np.array(from_im)
                        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

                        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                        face_detector = cv2.CascadeClassifier("face_detection.xml")
                        face_rects = face_detector.detectMultiScale(gray, 1.04, 5, minSize=(30, 30))
                        
                        result = cv_image
                        if face_rects != ():
                            idx = np.argmax(face_rects[:,2])
                            detect = face_rects[idx]
                            x, y, i, j = detect
                            x_pad = np.min(np.array([x, w-x-i, i//4]))
                            u_pad = np.min(np.array([y, j//2]))
                            d_pad = np.min(np.array([h-y-j, j//4]))
                            result = cv_image[y-u_pad:y+j+d_pad, x-x_pad:x+i+x_pad]

                        pil_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        from_im = Image.fromarray(pil_image)

                        w, h = from_im.size
                        if h / w > 1.15:
                            pad = (h - w) // 2
                            result = Image.new(from_im.mode, (h, h), (255, 255, 255))
                            result.paste(from_im, (pad, 0))
                            from_im = result
                        
                        os.makedirs(os.path.join(cropped_img_path, nton, TSn, AB, '3.Age'), exist_ok=True)
                        from_im.save(os.path.join(cropped_img_path, nton, TSn, AB, '3.Age', imgs))
                        #sys.exit()
                        pbar.update(1)
                print(os.path.join(img_folder_path, nton, TSn, AB))
pbar.close()
'''
################################################################################

### Code for cropping validation images ###
## To crop your individual image datasets, make your own crop code with below ##
################################################################################
for VSn in os.listdir(img_folder_path):
    for AB in os.listdir(os.path.join(img_folder_path, VSn)):
        for imgs in os.listdir(os.path.join(img_folder_path, VSn, AB, '3.Age')):
            if os.path.isfile(os.path.join(img_folder_path, VSn, AB, '3.Age', imgs)):
                from_im = Image.open(os.path.join(img_folder_path, VSn, AB, '3.Age', imgs))
                w, h = from_im.size
                np_image = np.array(from_im)
                cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                face_detector = cv2.CascadeClassifier("face_detection.xml")
                face_rects = face_detector.detectMultiScale(gray, 1.04, 5, minSize=(30, 30))
                
                result = cv_image
                if face_rects != ():
                    idx = np.argmax(face_rects[:,2])
                    detect = face_rects[idx]
                    x, y, i, j = detect
                    x_pad = np.min(np.array([x, w-x-i, i//4]))
                    u_pad = np.min(np.array([y, j//2]))
                    d_pad = np.min(np.array([h-y-j, j//4]))
                    result = cv_image[y-u_pad:y+j+d_pad, x-x_pad:x+i+x_pad]

                pil_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                from_im = Image.fromarray(pil_image)

                w, h = from_im.size
                if h / w > 1.15:
                    pad = (h - w) // 2
                    result = Image.new(from_im.mode, (h, h), (255, 255, 255))
                    result.paste(from_im, (pad, 0))
                    from_im = result
                
                os.makedirs(os.path.join(cropped_img_path, VSn, AB, '3.Age'), exist_ok=True)
                from_im.save(os.path.join(cropped_img_path, VSn, AB, '3.Age', imgs))
                #sys.exit()
                pbar.update(1)
        print(os.path.join(img_folder_path, VSn, AB))
pbar.close()
################################################################################
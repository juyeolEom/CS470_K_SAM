from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import cv2
import numpy as np


class InferenceDataset(Dataset):

    def __init__(self, root=None, paths_list=None, opts=None, transform=None, return_path=False):
        if paths_list is None:
            self.paths = sorted(data_utils.make_dataset(root))
        else:
            self.paths = data_utils.make_dataset_from_paths_list(paths_list)
        self.transform = transform
        self.opts = opts
        self.return_path = return_path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert(
            'RGB') if self.opts.label_nc == 0 else from_im.convert('L')

        ###       Here is the code for cropping the inference images        ###
        ###      Annotate it if you don't want to crop the input images      ###
        ########################################################################
        w, h = from_im.size
        np_image = np.array(from_im)
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier("face_detection.xml")
        face_rects = face_detector.detectMultiScale(
            gray, 1.04, 5, minSize=(30, 30))

        result = cv_image
        if face_rects != ():
            idx = np.argmax(face_rects[:, 2])
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
        ########################################################################

        if self.transform:
            from_im = self.transform(from_im)
        if self.return_path:
            return from_im, from_path
        else:
            return from_im

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os, shutil
import cv2
import numpy as np


DATA_PATH = '/opt/dkube/input/'
OUT_PATH = '/opt/dkube/output/'
IN_IMG_PATH = DATA_PATH + 'HPIA/Nucleoplasm_Cytosol_train/'
OUT_IMG_PATH = OUT_PATH + 'HPIA/Nucleoplasm_Cytosol/'

annot = pd.read_csv(DATA_PATH + 'HPIA/train.csv')

def open_rgby(path,id):
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id + '_' + color + '.png'), flags) for color in colors]
    img = np.stack(img, axis=-1)
    img = cv2.resize(img, (512, 512))
    return img

if not os.path.exists(OUT_IMG_PATH):
    os.makedirs(OUT_IMG_PATH)
j=0
ids = np.append(annot[annot['Target']=='0'].Id.values, annot[annot['Target']=='25'].Id.values)

for each_id in ids:
    temp_img = open_rgby(IN_IMG_PATH, each_id)
    cv2.imwrite(OUT_IMG_PATH + each_id + '.png', temp_img)
    j += 1
    if j%100==0:
        print('Finished Merging ' + str(j) + ' images')

shutil.copyfile(DATA_PATH + 'HPIA/train.csv', OUT_PATH + 'HPIA/train.csv')
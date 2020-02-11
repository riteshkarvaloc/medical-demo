import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os, shutil
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
import cv2
from random import shuffle

def build_annot(path, image_file, annot):
    img = cv2.imread(path + image_file)
    imgray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 40, 255, 0)
    contours, hierarchy =  cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    x_min = list()
    x_max = list()
    y_min = list()
    y_max = list()
    my_images = list()
    my_classes = list()
    for (i, c) in enumerate(cnts):    
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 30 and w < 100 and h > 30 and h < 100:
            if x-20>0:
                x_min.append(x-20)
            else:
                x_min.append(0)
            x_max.append(x + w + 20)
            if y-20>0:
                y_min.append(y-20)
            else:
                y_min.append(0)
            y_max.append(y + h + 20)
            my_images.append(path + image_file)
            my_classes.append(str(np.where(image_file in annot[annot['Target']=='0'].Id.values + '.png', 
                                           'Nucleoplasm', 'Cytosol')))
    
    annot_df = pd.DataFrame({'Image': my_images,
                             'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max,
                             'Class': my_classes})
    return annot_df


def annot_images(chunks, outdir, annot):
    annot_df = pd.DataFrame()
    j=0
    merged_imgs = os.listdir(outdir + 'Nucleoplasm_Cytosol/')
    for each_img in merged_imgs:
        annot_df = pd.concat([annot_df, build_annot(outdir + 'Nucleoplasm_Cytosol/', each_img, annot)], 
                            ignore_index=True)
    annot_df = annot_df.astype({'xmin': 'int32',
                                'xmax': 'int32',
                                'ymin': 'int32',
                                'ymax': 'int32'})
    annot_df.to_csv(outdir + 'annot.txt', header = False, index=False, sep=',')

def normalize_ratio(ratio, n_samples):
    if len(ratio) != 2:
        print("error: Ratio has less/more than three numbers")
        exit(1)
    if sum(ratio) <= 100:
        c_sizes = [(r/100)*n_samples for r in ratio]
        norm = []
        items_left = n_samples
        for cs in c_sizes:
            cr = cs / items_left
            norm.append(cr)
            items_left -= cs
        if sum(ratio) == 100:
            case = 1
        else:
            case = 2
        return norm, case
    elif sum(ratio) > 100:
        case = 3
        norm = [r/100 for r in ratio]
        return norm, case
    else:
        print("Ratio is not correct")
        exit(1)

def split_imgs(seed, files, case):
    chunks = []
    np.random.seed(seed)
    seeds = np.random.randint(1,45,3)
    i = 0
    if case == 1:
        for cr in s_ratio[:-1]:
            files, d = train_test_split(files, test_size=cr, random_state = seeds[i])
            chunks.append(d)
            i += 1
        chunks.append(files)
        return chunks
    elif case == 2:
        for cr in s_ratio:
            files, d = train_test_split(files, test_size=cr, random_state = seeds[i])
            chunks.append(d)
            i += 1
        return chunks
    elif case == 3:
        for cr in s_ratio:
            _ , d = train_test_split(files, test_size=cr, random_state = seeds[i])
            chunks.append(d)
            i += 1
        return chunks

def save_imgs(img_chunks, imgfolder, outputdir, annot):
    if not os.path.exists(outputdir[0] + 'Nucleoplasm_Cytosol'):
        os.makedirs(outputdir[0] + 'Nucleoplasm_Cytosol')
    for f in img_chunks[0]:
        shutil.copy(imgfolder + f, outputdir[0] + 'Nucleoplasm_Cytosol')
    annot_images(chunks[0], outputdir[0], annot)
    print("Train Data Created")

    if not os.path.exists(outputdir[1] + 'Nucleoplasm_Cytosol'):
        os.makedirs(outputdir[1] + 'Nucleoplasm_Cytosol')
    for f in img_chunks[1]:
        shutil.copy(imgfolder + f , outputdir[1] + 'Nucleoplasm_Cytosol')
    annot_images(chunks[1], outputdir[1], annot)
    print("Test Data Created")

if __name__== "__main__":
    parser = ArgumentParser(description="Split dataset into multiple chunks provided the ratio.\n"
                                        "Example: [python3 split.py  --ratio 10 10 80 --seed 13"
                                        " --outputdir data_splits/]", formatter_class=RawTextHelpFormatter)
    parser.add_argument("-r","--ratio", dest="ratio", default=[90,10],nargs='+',
                        help="dataset split ratio", type=int)
    parser.add_argument("-s", "--seed",dest="seed", type=int,default=13,
                        help="a seed value for random generator,"
                             " helps in regenerating the split",)
    parser.add_argument("-o","--outputdir", default='/opt/dkube/outputs/', dest = 'outdir', help="output folder path")
    args = parser.parse_args()

    DATA_DIR = '/home/dkube/work/workspace/' #"/opt/dkube/input"
    OUT_DIR = '/home/dkube/work/workspace/splits/'  #args.outdir
    TRAIN_DATA = OUT_DIR + 'train/'
    TEST_DATA = OUT_DIR + 'test/'
    annot = pd.read_csv(DATA_DIR + 'train.csv')
    imgfolder = DATA_DIR + '/Nucleoplasm_Cytosol/'
    img_names = os.listdir(imgfolder)
    shuffle(img_names)
    n_samples = len(img_names)
    s_ratio, case = normalize_ratio(args.ratio, n_samples)
    chunks = split_imgs(args.seed, img_names, case)
    save_imgs(chunks, imgfolder, [TRAIN_DATA, TEST_DATA], annot)
import os
import argparse
import cv2
import numpy as np
import json 
import sys 
import face_preprocess
import glob


def get_lmk_from_file(image,lmk_path):
    """
    Gets landmarks for a particular image in rfw dataset, given lmk file and image 
    :param image:  image to extract lmk
    :param lmk_path: path to lmk file
    """

    with open(lmk_path) as f:
        lines = f.readlines()

        for line in lines:
            line_list = line.split('\t')
            line_image = line_list[0].split('/')[-1]
            
            if line_image == image:

                if '\n' in line_list[-1]:
                    line_list[-1] = line_list[-1][:-1]

                lmks = np.array([[line_list[2],line_list[3]],
                        [line_list[4],line_list[5]],
                        [line_list[6],line_list[7]],
                        [line_list[8],line_list[9]],
                        [line_list[10],line_list[11]]])

                return lmks
            

def get_bb_from_file(image,lst_path,width,height):
    """
    Gets bounding_box for a particular image in rfw dataset, given lmk file and image 
    :param image:  image to extract lmk
    :param lst_path: path to lst file
    :param width: width of image
    :param height: height of image
    """

    with open(lst_path) as f:
        lines = f.readlines()


        for line in lines:
            line_list = line.split('\t')
            line_image = line_list[0].split('/')[-1]

            if line_image == image:
                _bbox = np.array(line_list[1:],dtype=float)

                return _bbox
            
    x, y, w, h = 0, 0, width, height
    _bbox = [x, y, w, h]
    return _bbox

if __name__ == '__main__':

    ### user arguments
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--dataset", type=str, default="D:../RFW/test/data/Indian/") 
    parse.add_argument("--lst", type=str, default="./bounding_boxes/Indian/lst_.txt")
    parse.add_argument("--lmk", type=str, default="./lmks/Indian_lmk.txt")
    parse.add_argument("--output", type=str, default="D:../RFW/test_aligned/data/Indian/")
    parse.add_argument('--image-size', type=str, help='Image size (height, width) in pixels.', default='112,112') 
    args = parse.parse_args()


    files = glob.glob(args.dataset + '*/*')

    for file in files:
        file = file.replace("\\","/")
        identity, image = file.split('/')[-2],  file.split('/')[-1]
        img = cv2.imread(args.dataset + identity + '/' +image)
        height, width, _ = np.asarray(img.shape)

        _bbox = np.array(get_bb_from_file(image,args.lst,width,height),dtype=float)
        _lmks = get_lmk_from_file(image, args.lmk)

        img = cv2.imread(args.dataset + identity + '/' +image)

        warped = face_preprocess.preprocess(img, bbox=_bbox,landmark=_lmks ,image_size=args.image_size)


        

        if not os.path.exists(args.output + identity ):
            os.makedirs(args.output + identity)


        cv2.imwrite(args.output + identity + '/' +image, warped)



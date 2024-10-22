#imports
import cv2
from PIL import Image
import numpy as np
import argparse


import glob
import os


def get_relative_files_with_extension_recursively(folder_path, extension):
    # Using glob to recursively find files with a specific extension and return relative paths
    return [os.path.relpath(f, folder_path) for f in glob.glob(os.path.join(folder_path, f'**/*.{extension}'), recursive=True) if os.path.isfile(f)]


def create_folder_if_needed(file_path):
    # Extract the directory path from the file path
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        # Create the directories if they don't exist
        os.makedirs(directory)



if __name__ == '__main__':



    ### user arguments
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--path_to_occlusion_folder", type=str, default = '../Occlusions/RFW1_occlusions_centered/African/')
    parse.add_argument("--path_to_rfw_folder", type=str, default = '../RFW/test_aligned/data/African/')
    parse.add_argument("--save_path",type=str,default = '../RFW1_v2/African/')

    args = parse.parse_args()


    #get all occlusion relative paths
    relative_paths = get_relative_files_with_extension_recursively(args.path_to_occlusion_folder,'png')


    for rel_path in relative_paths:

        path_to_occ = args.path_to_occlusion_folder + rel_path
        path_to_base = args.path_to_rfw_folder + rel_path[:-4] + '.jpg'

        
        # Load the occ
        occlusion = Image.open(path_to_occ) #removed RGBA conversion


        # Load the base image
        base_image  = Image.open(path_to_base)

        create_folder_if_needed(args.save_path + rel_path[:-4])

        result = base_image.copy()

        width, height = occlusion.size

        for i in range(width):
            for j in range(height):
                
                occlusion_pixel = occlusion.getpixel((i, j))

                if occlusion_pixel[3]>0:
                    result.putpixel((i, j), occlusion_pixel[:3])



        result.save(args.save_path + rel_path[:-4]+ '.jpg')



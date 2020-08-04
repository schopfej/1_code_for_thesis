import os
import re
import shutil
import os, shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', default="/home/jonathan/Documents/try_error_grouping_eye/test_pers_6/", type=str, help='PATH to Folder Train and Val')
parser.add_argument('--moveto', default="/home/jonathan/Documents/try_error_grouping_eye/radius/", type=str, help='Move coord in this directory')

args = parser.parse_args()


path = args.path 
moveto = args.moveto
#################################################
## Here regex is used ###########################
#################################################

pattern = re.compile(r"_\-?[0-3]_\-?[0-3]_")

files = os.listdir(path)
files.sort()
for filename in files:
    print(filename)
    if pattern.search(filename):
        try:

            src = path+filename
            dst = moveto+filename
            shutil.move(src,dst)

        except EnvironmentError:
            print("Does not work")
            pass

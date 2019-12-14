import os



# This is for if it's necessary to rename files. The file path here is the one from my machine; it will have to be adapted to run on another machine
# directory = "/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/Raven/"
# for filename in os.listdir(directory):
# 	dst = "raven"+filename
# 	src=directory+filename
# 	dst=directory+dst
# 	os.rename(src, dst)

import glob
import shutil
import random


def splitem(src, dest, case):
    files = glob.glob(src)
    random.shuffle(files)
    test_len = int(len(files) * 0.05)
    train_len = int(len(files) * 0.62)
    val_len = len(files) - train_len - test_len

    print(len(files))
    print(train_len)
    print(val_len)

    for i, file in enumerate(files):
        if i < test_len:
            shutil.copyfile(file, f"/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/case{case}/test/{dest}/{os.path.basename(file)}")
        if i < train_len:
            shutil.copyfile(file, f"/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/case{case}/train/{dest}/{os.path.basename(file)}")
        else:
            shutil.copyfile(file, f"/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/case{case}/validate/{os.path.basename(file)}")


puff_src = "/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/Puffin/*.*"
puff_dest = "puffin"

rav_src = "/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/Raven/*.*"
rav_dest = "raven"



for i in range(15):
    splitem(puff_src, puff_dest, str(i))
    splitem(rav_src, rav_dest, str(i))
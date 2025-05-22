import os
import shutil

val_images_dir = "tiny-imagenet-200/val/images" #current validation image location
annotations_file = "tiny-imagenet-200/val/val_annotations.txt" #labels of each image
output_root = "tiny-imagenet-200/val" #where to put the class folders

os.makedirs(output_root, exist_ok=True)

with open(annotations_file, 'r') as f:
    for line in f:
        img_name, cls_label, *_ = line.strip().split('\t')
        src_path = os.path.join(val_images_dir, img_name)
        dest_dir = os.path.join(output_root, cls_label)
        dest_path = os.path.join(dest_dir, img_name)

        os.makedirs(dest_dir, exist_ok=True) #Create the class folder if it doesn't exist
        shutil.move(src_path, dest_path) #move the image in corresponding class folder

print("Done—validation images are now in val/<class_label>/")  

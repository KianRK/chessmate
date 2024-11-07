#!/usr/bin/env python3.6

# Execute script in annotation directory

import os

image_dir = os.listdir("/home/admin-krajabi/chessmate/images_upwards")

annotation_dir = os.listdir()

i = 0

for file in image_dir:
    # Crop ".jpg" from end of filename
    file = file[:-4]

    # Split filename at "a:" which is only present in variations
    # so that split[0] contains name of original image
    split_name = file.split("a:")

    # Annotation is has the same name as the image but with an ".xml" at the end
    annotation_name = splitname[0] +".xml"

    # New annotation will be .xml file with the name of the variation
    new_annotation = file + ".xml"

    # Check if variation does not already have a annotation in directory
    # and that the original image has one
    if(new_annotation not in dir and annotation_name in dir):
        
        # Prepare file name strings to be substituted in .xml file
        old_image_name = splitname[0] + ".jpg"
        new_image_name = file + ".jpg"

        # Read content from original annotation file
        with open(annotation_name, "r") as file1:
            lines = file1.readlines()
        
        # Original image name in line 3 has to be replaced with name of
        # image variation
        lines[2] = lines[2].replace(old_image_name, new_image_name)
        
        # Create new annotation file and write adjusted content to it
        with open(new_annotation, "w") as file1:
            for line in lines:
                file1.write(line)



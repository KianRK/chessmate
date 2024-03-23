#!/usr/bin/env python3.6

import numpy as np
import cv2
import sys
import argparse
import os

#The purpose of this script is to iterate over a directory of (grayscale)images
#and create several copies with randomized contrast(alpha) and brightness(beta) values. 
#to center the values in both positive and negative direction in a similar distance around the 
# original value, we combine two normal distributions for each value.
# We choose mean and standard deviation so that the combined distributions form a new distribution
# with two modes around the respective means

VARIATIONS_PER_IMAGE = 5
ALPHA_DEFAULT = 1
ALPHA_DISTANCE = 0.15
ALPHA_STD_DEV = 0.05
BETA_DEFAULT = 0
BETA_DISTANCE = 20
BETA_STD_DEV = 5

def main():

    # Creating possibility to define an output path for captured images
    parser = argparse.ArgumentParser(description="randomizing contrast and brightness values of images in directory")
    parser.add_argument("imagedir", nargs='?', default="")
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.imagedir) and not args.imagedir=="":
        print("Output directory {} does not exist.".format(args.outputpath))
        sys.exit()

    image_files = os.listdir(args.imagedir)
    os.chdir(args.imagedir)
    samples_per_value = len(image_files) * VARIATIONS_PER_IMAGE

    alpha_values = create_multimodal_probability_distribution(ALPHA_DEFAULT, ALPHA_DISTANCE, ALPHA_STD_DEV, samples_per_value)
    beta_values = create_multimodal_probability_distribution(BETA_DEFAULT, BETA_DISTANCE, BETA_STD_DEV, samples_per_value)

    j = 0
    for i, file in enumerate(image_files):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for j in range(VARIATIONS_PER_IMAGE):
            index = i * VARIATIONS_PER_IMAGE + j
            alpha_string = "{:.2f}".format(alpha_values[index])
            alpha_string= alpha_string.replace(".","_")
            beta_string = "{:.0f}".format(beta_values[index])
            image_path = os.path.join(args.imagedir,file[:-4]+"a:{},b:{}".format(alpha_string,beta_string)+".jpg")
            variation = cv2.convertScaleAbs(image, alpha=alpha_values[index], beta=beta_values[index])
            print("image path: {}".format(image_path))
            variation = cv2.cvtColor(variation, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(image_path, variation)


def create_multimodal_probability_distribution(default_value, distance_from_default, std_deviation, sample_size):

    low_mode = default_value-distance_from_default
    high_mode = default_value+distance_from_default
    distributions = [
        {"type": np.random.normal, "kwargs": {"loc": low_mode , "scale": std_deviation}},
        {"type": np.random.normal, "kwargs": {"loc": high_mode , "scale": std_deviation}}, 
    ]
    coefficients = np.array([0.5, 0.5])

    #only necessary to normalize if coefficients dont add up to 1
    #coefficients /= coefficients.sum()

    num_distr = len(distributions)
    data = np.zeros((sample_size, num_distr))
    for idx, distr in enumerate(distributions):
        data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
        for value in data[:, idx]:
            if abs(default_value-value) > abs(distance_from_default+3*std_deviation):
                value = default_value
    random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
    sample = data[np.arange(sample_size), random_idx]

    return sample

    
if __name__ == "__main__":
    main()

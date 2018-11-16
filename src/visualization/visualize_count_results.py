#!/usr/bin/env python

import sys
import numpy as np
import cv2
from skimage import io
import seaborn as sns

def get_rgb_list(sns_palette):
    
    rgb_list = []

    for color in sns_palette:
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)
        rgb_byte = [r, g, b]
        rgb_list.append(rgb_byte)
    
    return rgb_list


def visualize_count_results(
    count_results, background_image, car_max, 
    cmap='Reds', line_rgb=[255, 0, 0], line_thickness=5, alpha=0.6, min_car_to_show=1):

    visualization_result = background_image.copy()
    overlay = background_image.copy()

    sns_palette = sns.color_palette(cmap, n_colors=car_max + 1)
    rgb_list = get_rgb_list(sns_palette)

    for count_result in count_results:
        
        position = count_result['position']
        top, left, bottom, right = position['top'], position['left'], position['bottom'], position['right']

        cars = count_result['cars']
        cars_counted = cars['counted']

        if cars_counted < min_car_to_show:
            continue
        
        overlay[top:bottom, left:right] = rgb_list[cars_counted]
        cv2.rectangle(overlay, (left, top), (right, bottom), line_rgb[::-1], thickness=line_thickness)
    
    cv2.addWeighted(overlay, alpha, visualization_result, 1 - alpha, 0, visualization_result)

    return visualization_result
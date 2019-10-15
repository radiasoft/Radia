#!/usr/bin/python

import numpy as np


def compute_Br(Bx, By, theta):
    Br = Bx * np.cos(theta) + By * np.sin(theta)
    return Br
    

def compute_modes(Br, theta, mode_list):
    ## Uses riemann sum to compute the modes from the field on a circle
    ## Mode list is the list of mode indices you wish to compute 
    
    d_theta = np.mean(np.diff(theta))
    Bm = []
    
    for i in range(0,len(mode_list)):
        Bm.append(1. / np.pi * np.sum(Br * np.sin(mode_list[i] * theta)) * d_theta)
    
    return Bm

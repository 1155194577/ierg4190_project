import numpy as np
from skimage.feature import hog as skhog
## TODO implement feature extractors : HOG, pixel (complu)
## TODO optional  : implement color_hist , bovw, pyramid....  
## TODO try CNN? 

def _extract_pixels(x): 
    ## return the flatten vector of the image
    return np.asarray(x, dtype=np.float32).reshape(len(x), -1)

def _extract_hog(x):
    ## ref => Histogram of Oriented Gradients (Dalal & Triggs, 2005)
    ## question asked : which directions edges point in? how strong are they in the local patch?
    ## find grad => find histogram => group cell into block + normalize => concat => final feature vector
    orientations = 9 
    pixels_per_cell = (8, 8) 
    cells_per_block = (2, 2) 
    channels_axis = -1 ## the last axis is the color channel axis, default is 3 for RGB images
    feature = []
    for i in range(len(x)):
        hog = skhog(x[i], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, channel_axis=channels_axis)
        feature.append(hog)
    return np.asarray(feature, dtype=np.float32).reshape(len(x), -1)

def extract(x, name="pixels"):
    ## extract feature from images. 
    assert len(x.shape) == 4, "Input must be a 4D array of shape (Number of images, Height, Width, Channels)" 
    ## supposedly should get N,32,32,3
    if name == "pixels":
        return _extract_pixels(x)
    
    elif name == "hog":
        return _extract_hog(x)
    else:
        raise ValueError(f"Unknown feature extractor: {name}")
    
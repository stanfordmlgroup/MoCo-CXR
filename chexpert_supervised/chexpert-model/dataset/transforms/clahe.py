import numpy as np
import cv2
from PIL import Image

class CLAHE(object):
    """ Apply CLAHE on a single image"""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = 2.0
        self.tile_grid_size = tile_grid_size

    def __call__(self, PIL_img, save = False):
        im_np = np.asarray(PIL_img)
        im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)

        # create a CLAHE object (Arguments are optional)
        clahe = cv2.createCLAHE(self.clip_limit, self.tile_grid_size)
        cl1 = clahe.apply(im_np)
        imaged = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(imaged)

        if save:
            # Saving images to display
            preimage.save(str(PIL_image) + "original.png")
            img.save(str(preimage) + "CLAHEd.png")
        return img

    # Not sure if the following is necessary
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

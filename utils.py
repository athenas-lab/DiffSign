import numpy as np
import cv2
from PIL import Image

import torch
from diffusers.utils import load_image


""" Image utility functions """

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
                        
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def getCannyImg(img):
    img = img.resize((1024,1024))
    img = np.array(img)


    #get canny image
    img = cv2.Canny(img, 100, 200)
    img = img[:, :, None]

    img = np.concatenate([img, img, img], axis=2)
    canny_img = Image.fromarray(img)

    return canny_img

def get_depth_map(img, depth_est):

    img = depth_est(img)["depth"]
    img = np.array(img)
    img = img[:, :, None]

    img = np.concatenate([img, img, img], axis=2)
    det_map = torch.from_numpy(img).float()/255.0

    depth_map = det_map.permute(2, 0, 1)

    return depth_map


def getDepthMap(img_path, depth_est, feat_ext):

    img = load_image(img_path)
    image = feat_ext(images=img, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
         depth_map = depth_est(image).predicted_depth
         depth_map = torch.nn.functional.interpolate(
                 depth_map.unsqueeze(1),
                 size=(1024, 1024),
                 mode="bicubic",
                 align_corners=False
              )

         
         depth_min = torch.amin(depth_map, dim=[1,2,3], keepdim=True)
         depth_max = torch.amax(depth_map, dim=[1,2,3], keepdim=True)
         depth_map = (depth_map - depth_min)/(depth_max - depth_min)
         image = torch.cat([depth_map]*3, dim=1)

         image= image.permute(0, 2, 3, 1).cpu().numpy()[0]
         image = Image.fromarray((image*255.0).clip(0, 255).astype(np.uint8))
         image.save("depth_map.png")
    return image


import numpy as np
from tqdm import tqdm
import cv2
import glob
import skimage
#import lpips
from PIL import Image
import torchvision.transforms as T
import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance as FID

""" 
Evaluating visual quality of generaed sign language videos:
   - Structural similarity computation between successive frames of a  synthetic sign language video.
   - FID computation of generated sign language video.
"""


def computeVideoSSIMFrames(fd):
  """ Use skimage to compute structural  similarity between frames as a measure of consistency """

  frames = sorted(glob.glob(fd + "/*.png"))
  prev_frame = cv2.imread(frames[0])
  im1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
  tot_ssim = 0.0

  for i, f in enumerate(frames[1:]):
    cur_frame = cv2.imread(f)
    im2 = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    ssim_val = skimage.metrics.structural_similarity(im1, im2, full=True)[0]

    tot_ssim += ssim_val
    prev_frame = cur_frame

  print(tot_ssim,len(frames)-1, tot_ssim/(len(frames)-1))    

  return


def preprocess(img_dir):
    """ Preprocess the  frames """

    import torchvision.transforms as T
    transform = T.Compose([
           T.Resize((512, 512)),
           T.PILToTensor(),

        ])
    img_list = sorted(glob.glob(img_dir + "/*.png"))[:1000]
    images = [Image.open(x) for x in img_list]
    images = [transform(image) for image in images]
    images = torch.stack(images, dim=0)
    #images = images.permute(0, 3, 1, 2)
    print(images.shape)

    return images
 
def computeFID(real_dir, fake_dir):

    real_im = preprocess(real_dir)
    fake_im = preprocess(fake_dir)

    fid_score = FID(normalize=True)
    fid_score.update(real_im, real=True)
    fid_score.update(fake_im, real=False)

    print("FID score", fid_score.compute())
    return



if __name__  == "__main__":
  
   #path to human signer and synthetic signer videos for computing directional similarity
   human_dir = "results/human_signer"
   synth_dir = "results/avatar2synth_img2img"
   

   #compute structural similarity across frames of synthetic signer video to 
   #assess the temporal inconsistency/jitter
   computeVideoSSIMFrames(synth_dir)

   #compute FID of synthetic signer videos wrt human signer video
   computeFID(human_dir, synth_dir)

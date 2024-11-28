from types import MethodType
import glob
import os
import numpy as np
import cv2
from PIL import Image

import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

from controlnet_aux import OpenposeDetector
from ip_adapter import IPAdapter

from diffusers.utils import load_image
import utils

""" 
    Using controlnet to transfer avatar poses to a synthetic signer 
    generated using  text prompts that are input to Stable Diffusion.
"""


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

    return image

class Image2ImageSigner:
      def __init__(self):
          """ Configure paths to pretrained modelss and Stable Diffusion settings """

          self.base_model_path = "runwayml/stable-diffusion-v1-5"
          self.vae_model_path = "stabilityai/sd-vae-ft-mse"
          self.image_encoder_path = "../models/image_encoder/"
          #self.ip_ckpt = "train_output/laion_400m/ip_adapter.bin"  #SK TRAIN
          self.ip_ckpt = "../models/ip-adapter_sd15.bin"
          self.device = torch.device("cuda:0")

          self.noise_scheduler = DDIMScheduler(
             num_train_timesteps=1000,
             beta_start=0.00085,
             beta_end=0.012,
             beta_schedule="scaled_linear",
             clip_sample=False,
             set_alpha_to_one=False,
             steps_offset=1,
            )
          self.vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16)


      def controlPoseCannyDepthSignAvatar(self):
          """ Use image prompt to control the appearance of the generated signer.
              Use a combination of pose, canny edge, depth map to transfer the 
              poses from avatar to synthetic signer.
          """

          p = 1 #pose control [1:include, 0:omit]
          c = 1 #canny edge control [1:include, 0:omit]
          d = 0 #depth control [1:include, 0:omit]

          controlnets = []
          if p == 1: #pose
             openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
             controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"
             controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
             controlnets.append(controlnet)
          if c == 1: #canny edge 
             canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                     torch_dtype=torch.float16, use_safetensors=True)
             controlnets.append(canny)
          if d == 1: #depth map
            depth_est = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            feat_ext = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            
            #load controlnet and stable diffusion
            depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                     variant="fp16",
                     torch_dtype=torch.float16, use_safetensors=True).to("cuda")
            controlnets.append(depth)
            
          #Load SD pipeline
          pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model_path,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            feature_extractor=None,
            safety_checker=None)

          img_size = (1024, 1024)
          ##set the path of the image prompt that will be  used to control the appearance of the synthetic signer
          image = Image.open("./sign_assets/cauc_man.png")
          image.resize(img_size)

          out_dir = "results/avatar2cauc_man"
          if not os.path.exists(out_dir):
             os.makedirs(out_dir) 

          #set the path of the folder that contains the avatar poses that will be  used 
          #as the conditioning inputs to control the poses of the synthetic signer when
          #generating the sign language video
          frame_dir = "sign_assets/half_body_avatar_sign_poses"
          frames = sorted(glob.glob(frame_dir + "/*.png"))

          #transfer poses from avatar videoto generate  synthetic signer video
          for i, f in enumerate(frames):
              pose_image = Image.open(f)

              ctrl_imgs = []
              if p == 1: #pose input
                 openpose_image = openpose(pose_image)
                 openpose_image.resize(img_size)
                 ctrl_imgs.append(openpose_image)   
              if c == 1: #canny edge input
                 can_img = utils.getCannyImg(pose_image)
                 can_img.resize(img_size)
                 ctrl_imgs.append(can_img)   
              if d == 1: #depth map input
                 depth_img = getDepthMap(pose_image, depth_est, feat_ext)
                 depth_img.resize(img_size)
                 ctrl_imgs.append(depth_img)   

              #use IPAdapter plugin for SD1.5  to generate the signer image based on image prompt 
              #and control inputs
              ip_model = IPAdapter(pipe, self.image_encoder_path, self.ip_ckpt, self.device)
              gen_img = ip_model.generate(pil_image=image, image=ctrl_imgs, width=1024, height=1024, num_samples=1, num_inference_steps=60, seed=42, scale=0.6)[0]
              #prompt="wearing black shirt", scale=0.5)[0]
              gen_img.save(os.path.join(out_dir, f.split("/")[-1])) 
              
          print(i, out_dir)
          return

      def controlPoseCannySignAvatarCustom(self):
          """ Generate signer using multimodal prompts (image + text) for customization """

          p = 1
          c = 1

          controlnets = []
          if p == 1: #pose  input
             openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
             controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"
             controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
             controlnets.append(controlnet)
          if c == 1: #canny edge input 
             canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                     torch_dtype=torch.float16, use_safetensors=True)
             controlnets.append(canny)

          #Load SD pipeline
          pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model_path,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            feature_extractor=None,
            safety_checker=None)

          #synthetic baseline images generated using  text-to-image model for appearance control
          signers = [os.path.join("sign_assets", x) for x in \
             ["boy.png", "woman.png"]]      
          sid = 2 #choose  the signer id for customization
          img_size = (1024, 1024)
          image = Image.open(signers[sid])
          image.resize(img_size)

          out_dir = "results/avatar2custom/" 
          out_dir = os.path.join(out_dir, os.path.splitext(signers[sid].split("/")[-1])[0])
          if not os.path.exists(out_dir):
             os.makedirs(out_dir) 

          frame_dir = "sign_assets/half_body_avatar_sign_poses"
          #transfer poses from selected avatar frames
          frames = [os.path.join(frame_dir, x) for x in ["0139.png", "0323.png", "0449.png", "0540.png"]]
          for i, f in enumerate(frames):
              pose_image = Image.open(f)
              print(f, pose_image.size)

              ctrl_imgs = []
              if p == 1:
                 openpose_image = openpose(pose_image)
                 openpose_image.resize(img_size)
                 ctrl_imgs.append(openpose_image)   
              if c == 1:
                 can_img = utils.getCannyImg(pose_image)
                 can_img.resize(img_size)
                 ctrl_imgs.append(can_img)   

              #use IPAdapter plugin for SD1.5  to generate the signer image based on image prompt 
              #and control inputs
              ip_model = IPAdapter(pipe, self.image_encoder_path, self.ip_ckpt, self.device)
              gen_img = ip_model.generate(pil_image=image, image=ctrl_imgs, width=1024, height=1024, num_samples=1, num_inference_steps=60, seed=42,
              #text prompt  for customizing the base signer
              prompt="wearing glasses with thick frames", scale=0.6)[0] 
              #prompt="wearing gray half sleeved shirt with black coat", scale=0.4)[0]
              gen_img.save(os.path.join(out_dir, f.split("/")[-1])) 
              
          print(out_dir)
          return

if __name__ == "__main__":

   adapter = IPAdapterSD15Control()
   adapter.controlPoseCannyDepthSignAvatar()

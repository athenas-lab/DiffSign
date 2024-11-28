import os
import numpy as np
import torch
import glob

import cv2
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler 
from diffusers import StableDiffusionControlNetImg2ImgPipeline
from diffusers import StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionXLControlNetPipeline, AutoencoderKL

from diffusers.utils import load_image
from transformers import pipeline
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

from controlnet_aux import OpenposeDetector 

""" 
    Using controlnet to transfer avatar poses to a synthetic signer 
    generated using  text prompts that are input to Stable Diffusion.
"""


#base_dir = "results/controlnet"

def getCannyImg(img_src):

    #download an image
    img = load_image(img_src)
    img = np.array(img)


    #get canny image
    img = cv2.Canny(img, 100, 200)
    img = img[:, :, None]

    img = np.concatenate([img, img, img], axis=2)
    canny_img = Image.fromarray(img)

    return canny_img


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


def controlPoseCannyDepthSign():

    """ 
    Extract  pose,  canny edge, and depth from rendered avatar (video) frames.
    Use these as control inputs to transfer the avatar poses to a synthetic signer using ControlNet.
    Generate a synthetic signer using text prompts to control the appearance using Stable Diffusion.
    """

    #get canny (edge) from source avatar image
    src = "sign_assets/half_body_avatar_sign_poses/0001.png"
    canny_img = getCannyImg(src)
    #canny_img = canny_img.resize((1024,1024))

    #detect pose from image using openpose
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose_img = load_image(src)
    pose_img = openpose(pose_img).resize((1024, 1024))

    #depth map
    depth_est = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feat_ext = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    depth_img = getDepthMap(src, depth_est, feat_ext)

    #load controlnet and stable diffusion
    depth = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",
             variant="fp16",
             torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    pose =  ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0",
                 torch_dtype=torch.float16) 
    canny = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0",
                 torch_dtype=torch.float16, use_safetensors=True)
    #controlnets = [pose, canny, depth] #depth map does not have much impact
    controlnets = [pose, canny]

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                          "stabilityai/stable-diffusion-xl-base-1.0", 
                          controlnet=controlnets, 
                          vae = vae,
                          torch_dtype=torch.float16,
                          use_safetensors=True).to("cuda")


    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt="A professional young Caucasian man with beard wearing white shirt, photorealistic, good lighting"
    neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    gen = torch.manual_seed(40)

    #images = [pose_img, canny_img, depth_img]
    #scales = controlnet_conditioning_scale=[1.0, 0.8, 0.5]

    #images = [canny_img, depth_img]
    #scales = controlnet_conditioning_scale=[0.8, 0.5]

    images = [pose_img, canny_img]
    scales = controlnet_conditioning_scale=[1.0, 0.8]

    #generate image with controlnet conditioning
    gen_img = pipe(prompt,
             negative_prompt=neg_prompt,
             image=images,
             num_inference_steps=50,
             generator=gen,
             num_images_per_prompt=3,
             #guidance_scale=3.0,
             #guess_mode=True,
             controlnet_conditioning_scale=scales
             ).images

    out_dir = "sdxl_pose_canny_cauc_man_signer"

    out_path = os.path.join(base_dir, out_dir, "0001_0.png")
    gen_img[0].save(out_path)
    out_path = os.path.join(base_dir, out_dir, "0001_1.png")
    gen_img[1].save(out_path)
    out_path = os.path.join(base_dir, out_dir, "0001_2.png")
    gen_img[2].save(out_path)

    return



if __name__ == "__main__": 

   controlPoseCannyDepthSign()


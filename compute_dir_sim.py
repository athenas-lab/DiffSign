import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import glob
from PIL import Image

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor
)

""" 
    Consider the pose transfer from human to synthetic signer as a style transfer problem
    and compute the directional similarity using CLIP to encode text and images.
"""

class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity
    

def computeDirSim():
    """ 
    The pose transfer from human to synthetic signer is treated as an image editing problem.
    Directional similarity measures how well the change in text prompt aligns with the
    change in the generated video with respect to the original human signer video. 
    """

    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
    scores = []

    #path to human signer and synthetic signer videos for computing directional similarity
    human_dir = "results/human_signer"
    synth_dir = "results/avatar2synth_img2img"


    input_images = []
    edited_images = []
    #human signer video 
    real_list = sorted(glob.glob(human_dir + "/*.png"))
    #synthetic videos
    fake_list = sorted(glob.glob(text_dir + "/*.png"))
    print(len(real_list), len(fake_list), real_list[0], fake_list[0])

    for i in range(len(real_list)):

        #caption/text description for original human  signer video
        original_image = Image.open(real_list[i])
        original_caption = "a female sign language signer"
        edited_image = Image.open(fake_list[i])

        #caption/text description for generated synthetic  signer video. 
        #This  represents the editing instruction.
        modified_caption = "a young Caucasian male sign language signer who has beard and is wearing a white shirt" 

        #compute the directional similarity score frame-by-frame for aaveraging
        similarity_score = dir_similarity(original_image, edited_image, original_caption, modified_caption)
        scores.append(float(similarity_score.detach().cpu()))

    #output the mean score
    print(f"CLIP directional similarity: {np.mean(scores)}") 

    return

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)     

    computeDirSim()

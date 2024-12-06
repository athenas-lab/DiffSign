# DiffSign 


## [DiffSign: AI-Assisted Generation of Customizable Sign Language Videos with Enhanced Realism](https://arxiv.org/abs/2412.03878)
#### Proceedings of [ECCV Workshop on Assistive Computer Vision and Robotics (ACVR), 2024](https://iplab.dmi.unict.it/acvr2024/)
This paper combines a parametric model and a generative model to generate synthetic signing videos 
in which the signer appearance can be customized in a zero-shot manner based on an image or
text prompt. The parametric model is used to retarget the signing poses with high fidelity while a diffusion
model is used to control the appearance of the synthetic signer. 
This repo provides the implementation of the generative phase. For retargeting the signing poses
from human signing videos to a 3D avatar, we used a pretrained SMPLify-X model and rendered the 
3D mesh into video frames using Blender. Please refer to the paper for further details.


!["Synthetic signer generation pipeline"](images/pipeline.jpg?raw=true)

### Installation of dependencies
- [Installation instructions for IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- Please download the pretrained IP-Adapter model as mentioned in the site.

- Stable Diffusion v1.5 (Got better results with v1.5 than later SD versions using IP-Adapter).

- In a virtual environment (conda, venv), execute:
```
pip install requirements.txt
```

### Training and testing
- The code uses pretrained IP-Adapter model, so no training is needed from scratch,
  but model can be fine-tuned on a few images using [DreamBooth](https://dreambooth.github.io/)
  (available in the diffusers package on HuggingFace) for personalization.
  
- In the code, the paths to the source videos and images are hard-coded. Please 
  change these to your paths before running the code.

- Generation using image prompt or multimodal prompt (image +text): 
```
python gen_diff_signer_image_prompt.py
```

- Generation using only text prompt : 
```
python gen_diff_signer_text_prompt.py
```

### Evaluation of the generated video as described in the paper:
  - Visual quality (SSIM, FID metrics): ```python compute_vis_quality.py```
  - Directional similarity: ```python compute_dir_sim.py```



## References  and Acknowledgments
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/index)


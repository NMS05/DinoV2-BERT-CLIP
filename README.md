# DinoV2-BERT-CLIP

This repo provides an oversimplified implementation of the OpenAI's CLIP model without the bells and whistles. If you are looking for a simple starter pack to train a CLIP model and play around various architectures, you are at the right place. Some highlights of this implementation are,

+ **Training Dataset** = MS-COCO 2017
+ **Vision Encoder** = Pre-trained DinoV2 ViT-B16 model from Meta.
+ **Text Encoder** = Pre-trained Distill BERT model from Huggingface.

## Results (Raw Similarity Scores)
<img src="https://github.com/NMS05/DinoV2-BERT-CLIP/blob/main/imgs/dog.png" width="400" height="400">
<img src="https://github.com/NMS05/DinoV2-BERT-CLIP/blob/main/imgs/clip_result.png" width="600" height="150">
<img src="https://github.com/NMS05/DinoV2-BERT-CLIP/blob/main/imgs/merlion.png" width="400" height="400">
<img src="https://github.com/NMS05/DinoV2-BERT-CLIP/blob/main/imgs/clip_result2.png" width="600" height="100">

## Directory Structure

+ **data/**
  - image_caption_data.py = a PyTorch Dataset class for MS-COCO that retuns a Image and its (bert) tokenized caption as a tensor.
  - utils.py = preprocesses image and caption for inference.
+ **data/**
  - model.py = contains the CLIP model with linear projection and the CLIP loss function.
+ train_clip.py = train the CLIP model and save weights.

## References
+ [Open_CLIP](https://github.com/mlfoundations/open_clip/tree/main/src)

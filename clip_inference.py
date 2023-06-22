import torch
import torch.nn as nn

from data.utils import prepare_data
from model.clip import clip_dinov2_bert

from PIL import Image
import requests


"""
prepare model
"""
device = torch.device("cuda:0")
model = clip_dinov2_bert()
model_weights = model.state_dict()

saved_weights = torch.load('clip.pth')
for name in model_weights:
    model_weights[name] = saved_weights['module.' + name]
model.load_state_dict(model_weights)
model.to(device)

print("\n\n\n===================================================================================\n\n")


"""
prepare data and get feature projections
"""
with torch.no_grad():
    process = prepare_data()
    img_link = "https://images.saymedia-content.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cq_auto:eco%2Cw_1200/MTk2MjE0MTk2MDU5MjUyMjUx/the-risks-of-playing-keep-away-games-with-your-dog.png"
    # img_link = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"
    image = Image.open(requests.get(img_link, stream=True).raw).convert('RGB')
    image = process.preprocess_image(image).unsqueeze(0)
    image = image.to(device)
    image_projection = model.forward_images(image)
    # print(image_projection.shape)

    captions = [
        'a dog playing with a red ball',
        'a picture of an airplane',
        'a dog running in the park'
    ]
    # captions = [
    #     'picture of the merlion in Singapore',
    #     'the empire state building in New York',
    # ]
    text_projections = []
    for cap in captions:
        input_ids, attention_mask = process.preprocess_text(cap)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_proj = model.forward_text(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        text_projections.append(text_proj.squeeze(0))
    text_projections = torch.stack(text_projections,dim=0)
    # print(text_projections.shape)


"""
get similarity scores
"""
similarity_scores = image_projection @ text_projections.T
similarity_scores = similarity_scores.cpu().numpy()

for i in range(len(captions)):
    print("\t",captions[i]," = ",similarity_scores[0][i],"\n")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

"""
CLIP model
"""  
class clip_dinov2_bert(torch.nn.Module):
    def __init__(self):
        super(clip_dinov2_bert, self).__init__()

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dino_projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 512),
        )
        
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 512),
        )
        
    def forward_images(self, images):
        dino_proj = self.dino_projection(self.dinov2(images))
        return F.normalize(dino_proj, dim=-1)
    
    def forward_text(self, input_ids, attention_mask):
        bert_out= self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_out['last_hidden_state'][:,0,:] # [CLS] tokens at index 0
        bert_proj = self.bert_projection(bert_cls)
        return F.normalize(bert_proj, dim=-1)
    
    def forward(self, images, input_ids, attention_mask):
        dino_cls = self.dinov2(images)
        dino_proj = self.dino_projection(dino_cls)

        bert_out= self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_out['last_hidden_state'][:,0,:] # [CLS] tokens at index 0
        bert_proj = self.bert_projection(bert_cls)

        return F.normalize(dino_proj, dim=-1), F.normalize(bert_proj, dim=-1)
    
"""
CLIP loss or Image-Text-Contrastive loss. Refer https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py 
"""
class ClipLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.logit_scale = 1.0/temperature

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features):
        logits_per_image = self.logit_scale * image_features @ text_features.T
        logits_per_text = self.logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss
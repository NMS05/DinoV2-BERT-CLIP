from torchvision.transforms import transforms
from transformers import DistilBertTokenizer

class prepare_data():
    def __init__(self,):
        self.val_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def preprocess_image(self, image):
        return self.val_transform(image)

    def preprocess_text(self,caption):
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=25,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask
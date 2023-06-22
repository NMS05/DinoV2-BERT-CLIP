from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pycocotools.coco import COCO
from PIL import Image
from transformers import DistilBertTokenizer

class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, apply_transform=False):

        # chunk for original COCO dataloader
        self.root_dir = root_dir
        self.train_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ColorJitter(brightness=.3, hue=.2),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.val_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.apply_transform = apply_transform
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # chunk for BERT tokenization
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        
        # chunk for original COCO dataloader
        coco = self.coco
        img_id = self.ids[index]
        caption = coco.imgToAnns[img_id][0]['caption'] # a python string
        # some captions have fullstop but others dont
        # so, remove '.' first, make lower case and finally add '.'
        caption = caption.strip('.').lower() + '.'
        # load and transform image
        img_info = coco.loadImgs(img_id)[0]
        img_path = f"{self.root_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')

        if self.apply_transform == True:
            image = self.train_transform(image) # tensor of shape = [3, H, W]
        else:
            image = self.val_transform(image)

        # chunk for BERT tokenization
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

        return image, input_ids, attention_mask

# # Define the paths to the dataset and annotations
# data_dir = "MSCOCO/val2017/"
# annotation_file = "MSCOCO/annotations/captions_val2017.json"

# # Create the dataset and dataloader
# coco_dataset = CocoDataset(data_dir, annotation_file)
# print(len(coco_dataset))
# coco_loader = DataLoader(coco_dataset, batch_size=4, shuffle=True)


# for image, input_ids, attention_mask in coco_loader:
#     print(image.shape, input_ids.shape, attention_mask.shape)
#     print(input_ids)
#     break
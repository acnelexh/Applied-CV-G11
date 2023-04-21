import torch.utils.data as data
from pathlib import Path
import torch
import numpy as np
import torchvision
import clip
from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPVisionModel
from template import imagenet_templates

class ImageTokenDataset(data.Dataset):
    '''
    Dataset that uses clip image encoder to encode image into tokens
    '''
    def __init__(self,
                image_dir: Path,
                device='cpu',
                vgg_transform=None):
        super(ImageTokenDataset, self).__init__()
        self.image_dir = image_dir
        self.image_processor = CLIPImageProcessor()
        self.device = device
        self.vgg_transform = vgg_transform

        self.images = [f for f in self.image_dir.glob('*')]

    def __getitem__(self, index):
        '''
        Return processed images
        One is preprocessed by CLIPImageProcessor, one is by source transform
        ''' 
        raw_img = torchvision.io.read_image(str(self.images[index])).to(self.device)
        img = self.image_processor(raw_img)
        img = torch.tensor(np.array(img['pixel_values'])).squeeze(0).to(self.device)
        #source_img = torchvision.io.read_image(str(self.images[index]))
        vgg_img = self.vgg_transform(raw_img).squeeze(0)
        # for model, pathloss, vgg content loss
        return img, vgg_img

    def __len__(self):
        return len(self.images)


class RandomTextDataset(data.Dataset):
    '''
    Dataset that returns random text descript for style transfer
    '''
    def __init__(self, text=['fire', 'pencil', 'water']):
        super(RandomTextDataset, self).__init__()
        self.text = text
        
    def __getitem__(self, index):
        return self.text[index]
    
    def __len__(self):
        return len(self.text)

# Testing ground

def test_text_encoder():
    "https://discuss.huggingface.co/t/last-hidden-state-vs-pooler-output-in-clipvisionmodel/26281"
    "average pooling is better than last hidden state???"
    "why not use both?"
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    #input_prompts = [template.format("pencil style") for template in imagenet_templates]
    input_prompts = ['apple']
    inputs = tokenizer(input_prompts, padding=True, return_tensors="pt")
    outputs = text_encoder(**inputs)

    print(len(imagenet_templates))
    print(outputs['last_hidden_state'].shape)
    print(outputs['pooler_output'].shape)
    print(outputs['pooler_output'] == outputs['last_hidden_state'][-1])


def test_text_loader():
    text_dataset = RandomTextDataset()
    print(text_dataset[0]['cls_token'].shape)
    print(text_dataset[0]['average_pooling'].shape)

def test_image_loader():
    image_dataset = ImageTokenDataset(Path("input_content/"))
    image_dataloader = data.DataLoader(image_dataset, batch_size=4, shuffle=False, num_workers=0)
    for i, input in enumerate(image_dataloader):
        print(input)
        break

def test_image_encoder():
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPImageProcessor()
    img = torch.rand(4, 3, 224, 224)
    img = processor(img)
    img['pixel_values'] = torch.tensor(img['pixel_values'])
    output = model(**img)
    print(output.keys())
    l = torch.nn.Linear(768,512)
    o = l(output['last_hidden_state'])
    print(output['last_hidden_state'].shape)
    print(o.shape)

#test_text_encoder()
#test_text_loader()
#test_image_encoder()
#test_image_loader()
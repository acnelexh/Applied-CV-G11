from torchvision import models, transforms
import torch
import clip
from transformers import CLIPImageProcessor
from util.clip_utils import get_features

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# vgg = models.vgg19(pretrained=True).features
# vgg.to(device)

class VGGNormalizer():
    def __init__(self, device='cpu', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(1,-1,1,1).to(device)
        self.std = torch.tensor(std).view(1,-1,1,1).to(device)
        self.transform = transforms.Compose(
            [transforms.Resize(size=(224, 224))])
    
    def __call__(self, x):
        return self.transform((x-self.mean)/self.std)
    

def get_content_loss(input_image, output_image, vgg, device='cuda'):
    '''
    Calculate content loss
    '''
    # noramlize targets to 0-1
    output_image = output_image.reshape((output_image.shape[0], output_image.shape[1], -1))
    output_image -= torch.min(output_image, dim=2, keepdim=True)[0]
    output_image /= torch.max(output_image, dim=2, keepdim=True)[0]
    output_image = output_image.reshape((output_image.shape[0],output_image.shape[1], 224, 224))

    content_features = [] # list of dict
    
    for img in input_image:
        content_features.append(get_features(img, vgg))

    target_features = [] # dict of hidden state from vgg of images
    VGGNORM = VGGNormalizer(device)
    for img in output_image:
        target_features.append(get_features(VGGNORM(img), vgg))
    
    content_loss = 0
    for i in range(len(target_features)):
        content_loss += torch.mean((target_features[i]['conv4_2'] - content_features[i]['conv4_2']) ** 2)
        content_loss += torch.mean((target_features[i]['conv5_2'] - content_features[i]['conv5_2']) ** 2)

    return content_loss
        
def get_text_direction(source_text, style_text, model, device='cpu'):
    '''
    Calculate text direction
    '''
    source_text = clip.tokenize(source_text).to(device)
    source_text = model.encode_text(source_text)
    
    style_text = clip.tokenize(style_text).to(device)
    style_text = model.encode_text(style_text)
    
    text_direction = style_text - source_text
    text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)
    return text_direction
    
def encode_img(images, model, device='cpu', preprocess=True):
    '''use clip api to encode image into 512-dim vector'''
    if preprocess:
        preprocessor = CLIPImageProcessor(device=device) # turns out to be exactly the same as the one in clip
        image = preprocessor(images)
        image_features = model.encode_image(torch.tensor(image['pixel_values']).to(device))
    else:
        image_features = model.encode_image(images)
    return image_features

def get_patches(imgs, args):
    '''
    Generate patches
    '''
    cropper = transforms.Compose(
        [transforms.RandomCrop(args.crop_size)]
        )
    augment = transforms.Compose(
        [transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
         transforms.Resize(224)]
        )
    img_aug = []
    for target in imgs:
        for _ in range(args.num_crops):
            target_crop = cropper(target)
            target_crop = augment(target_crop)
            img_aug.append(target_crop)
    img_aug = torch.stack(img_aug, dim=0)
    
    return img_aug


def get_img_direction(input_img, output_img, args, model, patch=False):
    '''
    Calculate image direction
    '''
    if patch == True:
        output_img = get_patches(output_img, args)
        crop_features = encode_img(output_img, model, device=args.device) # (batch_size x num_crops) x 512
        crop_features = crop_features.reshape((args.batch_size, args.num_crops, -1))
        image_features = torch.sum(crop_features, dim=1).clone()
    else:
        image_features = encode_img(output_img, model, device=args.device) # batch_size x 512
        
    source_features = encode_img(input_img, model, device=args.device, preprocess=False)

    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    return img_direction

def get_patch_loss(img_direction, text_direction, args):
    '''
    Calculate patch loss
    '''
    tmp_loss = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
    tmp_loss[tmp_loss < args.thresh] = 0 # TODO: add args
    #tmp_loss = torch.randn(64)
    patch_loss = tmp_loss.mean()

    return patch_loss

def get_glob_loss(image_direction, text_direction):
    glob_loss = (1 - torch.cosine_similarity(image_direction, text_direction, dim=1)).mean()
    return glob_loss
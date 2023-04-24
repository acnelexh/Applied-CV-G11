import clip
import torch
from torchvision import transforms
from torch.nn.functional import mse_loss
from transformers import CLIPImageProcessor
from util.clip_utils import get_features
from template import imagenet_templates

class CLIPNormalizer():
    def __init__(self, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711],device='cuda'):
        self.transform = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)])
    
    def __call__(self, x) -> torch.Tensor:
        return self.transform(x)

class VGGNormalizer():
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)])
    
    def __call__(self, x) -> torch.Tensor:
        return self.transform(x)
    
def get_content_loss(input_image, output_image, vgg, device='cuda'):
    '''
    Calculate content loss
    '''
    # noramlize targets to 0-1
    normalizer = VGGNormalizer()
    input_image = normalizer(input_image)
    output_image = normalizer(output_image)

    # using mse for now
    return mse_loss(input_image, output_image)
    # content_features = [] # list of dict
    
    # for img in input_image:
    #     content_features.append(get_features(img, vgg))

    # target_features = [] # dict of hidden state from vgg of images
    # VGGNORM = VGGNormalizer(device)
    # for img in output_image:
    #     target_features.append(get_features(VGGNORM(img), vgg))
    
    # content_loss = 0
    # for i in range(len(target_features)):
    #     content_loss += torch.mean((target_features[i]['conv4_2'] - content_features[i]['conv4_2']) ** 2)
    #     content_loss += torch.mean((target_features[i]['conv5_2'] - content_features[i]['conv5_2']) ** 2)

    # return content_loss

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(*text) for template in templates]

def get_text_direction(source_text, style_text, model, device='cpu'):
    '''
    Calculate text direction
    '''
    source_text = compose_text_with_templates(source_text, imagenet_templates)
    #print("raw source text", source_text)
    source_text = clip.tokenize(source_text).to(device)
    source_text = model.encode_text(source_text)
    #print("source text before normalizing", source_text)
    source_text = source_text.mean(axis=0, keepdim=True)
    source_text /= source_text.norm(dim=-1, keepdim=True)

    #print("The source text shape is: ", source_text.shape)
    #print("source text after normalizing", source_text)
    #exit()
    style_text = compose_text_with_templates(style_text, imagenet_templates)
    style_text = clip.tokenize(style_text).to(device)
    style_text = model.encode_text(style_text)
    style_text = style_text.mean(axis=0, keepdim=True)
    style_text /= style_text.norm(dim=-1, keepdim=True)

    #print("style text shape", style_text.shape)
    #print("source text shape", source_text.shape)
    #print("The style text is: ", style_text)
    #print("The source_text is: ", source_text)

    text_direction = (style_text - source_text).repeat(64,1) #TODO: make image_features[0] (64) an argument to pass in?
    text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)
    #print("text direction shape is: ", text_direction.shape)
    #print("The text direction is: ")
    #print(text_direction)
    return text_direction
    
def encode_img(images, model):
    '''use clip api to encode image into 512-dim vector'''
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
    normalizer = CLIPNormalizer()
    if patch == True:
        output_image = get_patches(output_img, args)
        # output_image = output_image.reshape((output_image.shape[0], output_image.shape[1], -1))
        # output_image -= torch.min(output_image, dim=2, keepdim=True)[0]
        # output_image /= torch.max(output_image, dim=2, keepdim=True)[0]
        # output_image = output_image.reshape((output_image.shape[0],output_image.shape[1], 224, 224))
        crop_features = encode_img(normalizer(output_image), model) # (batch_size x num_crops) x 512
        crop_features = crop_features.reshape((args.batch_size, args.num_crops, -1))
        image_features = torch.sum(crop_features, dim=1).clone()
    else:
        image_features = encode_img(normalizer(output_img), model) # batch_size x 512
        
    source_features = encode_img(input_img, model)

    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    return img_direction

def get_patch_loss(img_direction, text_direction, args):
    '''
    Calculate patch loss
    '''
    tmp_loss = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
    tmp_loss[tmp_loss < args.thresh] = 0 
    patch_loss = tmp_loss.mean()

    return patch_loss

def get_glob_loss(image_direction, text_direction):
    glob_loss = (1 - torch.cosine_similarity(image_direction, text_direction, dim=1)).mean()
    return glob_loss

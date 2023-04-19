import argparse
import os
import torch
import torch.nn as nn
import clip
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms, models
from tqdm import tqdm
from pathlib import Path
import models.transformer as transformer
import models.StyTR  as StyTR 
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from dataset import FlatFolderDataset, ImageTokenDataset, RandomTextDataset

from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPVisionModel
from template import imagenet_templates
from util.clip_utils import get_features

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def img_normalize(image):
    # TODO
    device = 'cpu'
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def encode_img(image, image_processor, image_encoder):
    image = image_processor(image)
    image['pixel_values'] = torch.tensor(image['pixel_values']) #?
    image = image_encoder(**image)
    return image.last_hidden_state.squeeze(0)[1:,:] # (196,768)

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./input_content/', type=Path,   
                    help='Directory path to a batch of content images')
parser.add_argument('--style_texts', type=list[str], default=['fire','water'],
                    help='List of style texts')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint

# training options TODO: modify options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch16',
                        help="CLIP model to use for the encoder")
parser.add_argument('--lambda_tv', type=float, default=2e-3)
parser.add_argument('--lambda_patch', type=float, default=9000)
parser.add_argument('--lambda_dir', type=float, default=500)
parser.add_argument('--lambda_c', type=float, default=150)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--thresh', type=float, default=0.7)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--num_crops', type=int, default=64)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

# vgg = StyTR.vgg
# vgg.load_state_dict(torch.load(args.vgg))
# vgg = nn.Sequential(*list(vgg.children())[:44])

#embedding = StyTR.PatchEmbed()

#Trans = transformer.Transformer()
with torch.no_grad():
    network = StyTR.StyTrans(args.clip_model)
network.to(device)
network.train()
vgg = models.vgg19(pretrained=True).features
vgg.to(device)
for parameter in vgg.parameters():
    parameter.requires_grad_(False)


#network = nn.DataParallel(network, device_ids=[0,1]) # probably don't need it 
# content_tf = train_transform()
# style_tf = train_transform()
_, preprocess = clip.load("ViT-B/32", device=device)
content_dataset = ImageTokenDataset(
    args.content_dir,
    clip_model=args.clip_model,
    device=args.device,
    source_transform=preprocess)
style_dataset = RandomTextDataset(
    args.style_texts,
    clip_model=args.clip_model,
    device=args.device) #TODO: try multiple styles?

# probably shouldn't do this if not necessary, wasting mem
source = ["a photo"] * len(args.style_texts)
source_dataset = RandomTextDataset(source)

# returns image embedding (source_features)
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))

# TODO: check, returns text_features (text embedding), normalize?
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

source_iter = iter(data.DataLoader(
    source_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(source_dataset),
    num_workers=args.n_threads))
 

optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

# TODO: discuss with team: use clip_styler sched policy or don't change

if not os.path.exists(args.save_dir+"/test"):
    os.makedirs(args.save_dir+"/test")

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []

cropper = transforms.Compose([
    transforms.RandomCrop(args.crop_size)
])
augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])

num_crops = args.num_crops # TODO: add args
image_processor = CLIPImageProcessor()
image_encoder = CLIPVisionModel.from_pretrained(args.clip_model)
source_features = None #TODO: raw images, not embedding

for i in tqdm(range(args.max_iter)):

    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    content_images, source_images = next(content_iter) # TODO: should prob return both raw imgs and embeddings
    style_texts = next(style_iter)
    source_texts = next(source_iter)

    targets = network(content_images, style_texts)
    targets = targets.reshape((targets.shape[0], targets.shape[1], -1))
    targets -= torch.min(targets, dim=2, keepdim=True)[0]
    targets /= torch.max(targets, dim=2, keepdim=True)[0]
    targets = targets.reshape((targets.shape[0],targets.shape[1], 224, 224))

    content_features = [] # list of dict
    for i in content_images['pixel_values']:
        content_features.append(get_features(img_normalize(i), vgg)) # TODO: IMPORTANT, get_features needs raw images, not embeddings

    target_features = [] # dict of hidden state from vgg of images
    for i in targets:
        target_features.append(get_features(img_normalize(i), vgg)) 

    content_loss = 0
    for i in range(len(target_features)):
        content_loss += torch.mean((target_features[i]['conv4_2'] - content_features[i]['conv4_2']) ** 2)
        content_loss += torch.mean((target_features[i]['conv5_2'] - content_features[i]['conv5_2']) ** 2)

    patch_loss = 0
    img_aug = []
    print("targets shape", targets.shape)
    for target in targets:
        # t = target < 0
        # if False in t:
        #     print("target causing exit", target)
        #     exit()
        for n in range(num_crops):
            target_crop = cropper(target)
            target_crop = augment(target_crop)
            img_aug.append(target_crop)
    
    # patch loss  
    # content_images['last_hidden_state'][0].shape is 196x768
    crop_features = [encode_img(img_aug[i], image_processor, image_encoder) for i in range(len(img_aug))]
    img_aug_features = []
    # TODO: unfinished
    for i in range(args.batch_size):
        crop_cat = torch.zeros(img_aug_features[0].shape)
        
    img_aug_features = [encode_img(img_aug[i], image_processor, image_encoder) for i in range(len(img_aug))]
    #img_aug_features = torch.cat(img_aug_features) # prob cat and just single for loop below
    cat_features = torch.zeros(img_aug_features.shape)
    
    img_aug_features /= (img_aug_features.clone().norm(dim=-1, keepdim=True)) # TODO: check dim
    img_direction = torch.zeros(img_aug_features[0].shape)
    # TODO: img_aug_features[i] should be the sum of the tensors of the crops of image i in img_aug_features
    for i in range(args.batch_size):
        #for c in range(num_crops):
        img_direction += img_aug_features[i] - content_images['last_hidden_state'][i]
    #img_direction = img_aug_features - source_features # might need for loop when implement later, should provide indexing for source_features.
    
    text_direction = (style_texts['average_pooling'] - source_texts['average_pooling']).repeat(img_aug_features.size(0),1) # TODO: check dim
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    tmp_loss = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
    tmp_loss[tmp_loss < args.thresh] = 0 # TODO: add args
    patch_loss += tmp_loss.mean()

    # global loss
    glob_features = [encode_img(i, image_processor, image_encoder) for i in targets['pixel_values']] # TODO: should provide index to get embeddings.
    
    glob_direction = (glob_features - source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

    glob_loss = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    var_loss = get_image_prior_losses(targets) # total variation loss

    total_loss = args.lambda_patch * patch_loss + args.content_weight * content_loss + args.lambda_tv * var_loss + args.lambda_dir * glob_loss
    total_loss_epoch.append(total_loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    ####

    if i % 50 == 0:
        # output_name = '{:s}/test/{:s}{:s}'.format(
        #                 args.save_dir, str(i),".jpg"
        #             )
        # out = torch.cat((content_images,out),0)
        # out = torch.cat((style_images,out),0)
        # save_image(out, output_name)
        print("After %d criterions:" % i)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('patch loss: ', patch_loss.item())
        print('dir loss: ', glob_loss.item())
        print('TV loss: ', var_loss.item())


    # writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    # writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    # writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    # writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    # writer.add_scalar('total_loss', loss.sum().item(), i + 1)
    writer.add_scaler('Total loss: ', total_loss.item())
    writer.add_scalar('Content loss ', content_loss.item())
    writer.add_scalar('patch loss: ', patch_loss.item())
    writer.add_scalar('dir loss: ', glob_loss.item())
    writer.add_scalar('TV loss: ', var_loss.item())

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
                                                   
writer.close()



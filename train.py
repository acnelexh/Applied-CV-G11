import argparse
import os
import torch
import clip
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torchvision import transforms, models
from tqdm import tqdm
from pathlib import Path
import models.StyTR  as StyTR 
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from dataset import ImageTokenDataset, RandomTextDataset
from loss import get_content_loss, get_img_direction, get_text_direction, get_patch_loss, get_glob_loss
from transformers import CLIPImageProcessor, CLIPVisionModel
from util.clip_utils import get_features

class VGGNormalizer():
    def __init__(self, device='cpu', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(1,-1,1,1).to(device)
        self.std = torch.tensor(std).view(1,-1,1,1).to(device)
        self.transform = transforms.Compose(
            [transforms.Resize(size=(224, 224))])
    
    def __call__(self, x):
        return self.transform((x-self.mean)/self.std)
    
def encode_img(images, device='cpu'):
    '''use clip api to encode image into 512-dim vector'''
    model, _ = clip.load("ViT-B/32", device=device)
    preprocess = CLIPImageProcessor(device=device) # turns out to be exactly the same as the one in clip
    image = preprocess(images)
    image_features = model.encode_image(torch.tensor(image['pixel_values']).to(device))
    return image_features

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
parser.add_argument('--num_crops', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)


with torch.no_grad():
    network = StyTR.StyTrans(args.clip_model)
network.to(device)
network.train()
vgg = models.vgg19(pretrained=True).features
vgg.to(device)
for parameter in vgg.parameters():
    parameter.requires_grad_(False)


#network = nn.DataParallel(network, device_ids=[0,1]) # probably don't need it 

_, preprocess = clip.load("ViT-B/32", device=device)
content_dataset = ImageTokenDataset(
    args.content_dir,
    clip_model=args.clip_model,
    device=args.device,
    clip_transform=preprocess,
    vgg_transform=VGGNormalizer(args.device))    
style_dataset = RandomTextDataset(
    args.style_texts,
    clip_model=args.clip_model,
    device=args.device) #TODO: try multiple styles?

# probably shouldn't do this if not necessary, wasting mem
source = ["a photo"] * len(args.style_texts)
source_dataset = RandomTextDataset(
    source,
    clip_model=args.clip_model,
    device=args.device)

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

num_crops = args.num_crops # TODO: add args
image_processor = CLIPImageProcessor(device=args.device)
image_encoder = CLIPVisionModel.from_pretrained(args.clip_model)
source_features = None #TODO: raw images, not embedding

for iteration in tqdm(range(args.max_iter)):
    #warm up
    if iteration < 1e4:
        warmup_learning_rate(optimizer, iteration_count=iteration)
    else:
        adjust_learning_rate(optimizer, iteration_count=iteration)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    content_images, clip_images, vgg_images = next(content_iter) # TODO: should prob return both raw imgs and embeddings
    style_texts = next(style_iter)
    source_texts = next(source_iter)
    
    targets = network(content_images, style_texts)
    
    content_loss = get_content_loss(vgg_images, targets, device=args.device)
    
    img_direction = get_img_direction(clip_images, targets, args, patch=True)
    text_direction = get_text_direction(style_texts, source_texts)
    
    # patch loss 
    patch_loss = get_patch_loss(img_direction, text_direction, args)

    # global loss
    img_direction = get_img_direction(clip_images, targets, args, patch=False)
    glob_loss = get_glob_loss(img_direction, text_direction)

    #var_loss = get_image_prior_losses(targets) # total variation loss, should loop
    var_loss = 0
    for i in targets:
        img = i.unsqueeze(0)
        var_loss += get_image_prior_losses(img)

    total_loss = args.lambda_patch * patch_loss + args.content_weight * content_loss + args.lambda_tv * var_loss + args.lambda_dir * glob_loss
    total_loss_epoch.append(total_loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    ####

    if iteration % 50 == 0:
        print("After %d criterions:" % iteration)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('patch loss: ', patch_loss.item())
        print('dir loss: ', glob_loss.item())
        print('TV loss: ', var_loss.item())

    writer.add_scalar('Total loss: ', total_loss.item())
    writer.add_scalar('Content loss ', content_loss.item())
    writer.add_scalar('patch loss: ', patch_loss.item())
    writer.add_scalar('dir loss: ', glob_loss.item())
    writer.add_scalar('TV loss: ', var_loss.item())

    if (iteration + 1) % args.save_model_interval == 0 or (iteration + 1) == args.max_iter:
        state_dict = network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/iter_{:d}.pth'.format(args.save_dir,
                                                           iteration + 1))
                                                   
writer.close()



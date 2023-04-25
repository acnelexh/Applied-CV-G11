import os
import PIL
import clip
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import models.StyTR  as StyTR 
from datetime import datetime
import torch.utils.data as data
from torchvision import models, utils
from tensorboardX import SummaryWriter
from sampler import InfiniteSamplerWrapper
from dataset import ImageTokenDataset, RandomTextDataset
from loss import get_content_loss, get_img_direction, get_text_direction, get_patch_loss, get_glob_loss


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


def main(args):  
    save_dir = Path(args.save_dir) / f"{args.exp_name}" / datetime.now().strftime("%Y%m%d-%H%M%S")
    print('mkdir', save_dir, '...')
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = str(save_dir)

    # save training parameters
    with open(os.path.join(args.save_dir, 'train.sh'), 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('python train.py \\\n')
        for key, value in vars(args).items():
            f.write("   --{} {} \\\n".format(key, value))

    # logging and tensorboard writers
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    
    network = StyTR.StyTrans(args)
    network.to(args.device)
    network.train()
    vgg = models.vgg19(pretrained=True).features
    vgg.to(args.device)
    for parameter in vgg.parameters():
        parameter.requires_grad_(False)

    if args.clip_model == 'openai/clip-vit-base-patch16':
        model, _ = clip.load("ViT-B/16", device=args.device)
    elif args.clip_model == 'openai/clip-vit-base-patch32':
        model, _ = clip.load("ViT-B/32", device=args.device)
    elif args.clip_model == 'openai/clip-vit-large-patch14':
        model, _ = clip.load("ViT-L/14", device=args.device)
    else:
        raise ValueError("Invalid CLIP model: %s" % args.clip_model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    
    content_dataset = ImageTokenDataset(
        args.content_dir,
        device=args.device)
    
    style_dataset = RandomTextDataset(args.style_texts) #TODO: try multiple styles?

    # probably shouldn't do this if not necessary, wasting mem
    source_dataset = RandomTextDataset(args.source_texts)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.max_batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))

    # TODO: check, returns text_features (text embedding), normalize? 
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.max_batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))
    #print("The style iter is: \n")
    #print(style_iter)
    source_iter = iter(data.DataLoader(
        source_dataset, batch_size=args.max_batch_size,
        sampler=InfiniteSamplerWrapper(source_dataset),
        num_workers=args.n_threads))
    
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)


    total_loss_epoch = []

    # since gpu can only support batch size of 2
    # iterate thorugh the dataset, and update the model
    # when there is enough gradient

    update_every_n_iters = args.batch_size // args.max_batch_size

    for iteration in tqdm(range(args.max_iter)):
        #warm up
        if iteration < 1e4:
            warmup_learning_rate(optimizer, iteration_count=iteration)
        else:
            adjust_learning_rate(optimizer, iteration_count=iteration)

        # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
        content_images, raw_images = next(content_iter) # TODO: should prob return both raw imgs and embeddings
        style_texts = next(style_iter)
        source_texts = next(source_iter)
        #print("The style texts are: ")
        #print(style_texts)   
        targets = network(content_images, style_texts)
        
        content_loss = get_content_loss(raw_images, targets, vgg, device=args.device)
        
        img_direction = get_img_direction(content_images, targets, args, model, patch=True)
        text_direction = get_text_direction(source_texts, style_texts, model, args, device=args.device, glob=False)
        
        # patch loss 
        patch_loss = get_patch_loss(img_direction, text_direction, args)

        # global loss
        img_direction = get_img_direction(content_images, targets, args, model, patch=False)
        text_direction = get_text_direction(source_texts, style_texts, model, args, device=args.device, glob=True)
        glob_loss = get_glob_loss(img_direction, text_direction)

        #var_loss = get_image_prior_losses(targets) # total variation loss, should loop
        var_loss = 0
        for i in targets:
            img = i.unsqueeze(0)
            var_loss += get_image_prior_losses(img)

        total_loss = \
                args.lambda_patch * patch_loss \
                + args.lambda_c * content_loss \
                + args.lambda_tv * var_loss \
                + args.lambda_dir * glob_loss
        total_loss = total_loss / update_every_n_iters # normalize loss
        total_loss.backward()

        if iteration % update_every_n_iters == 0:
            optimizer.step()
            optimizer.zero_grad()

        if iteration % (100 * update_every_n_iters) == 0:
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

        
        if (iteration + 1) % (100 * update_every_n_iters) == 0:
            # save targets
            for idx, img in enumerate(targets):
                style_descrption = style_texts[idx]
                # normalize img to unit8
                img = img.detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img = (img * 255).astype(np.uint8)
                # save image
                PIL.Image.fromarray(img).save(Path(args.save_dir)/ f'iter_{iteration}_{idx}_{style_descrption}.png')

        if (iteration + 1) % (args.save_model_interval * update_every_n_iters) == 0 or (iteration + 1) == args.max_iter:
            state_dict = network.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/iter_{:d}.pth'.format(args.save_dir,
                                                            iteration + 1))
                                                    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', default='./input_content/', type=Path,   
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_texts', type=str, default='./input_style/style.txt',
                        help='txt of style texts')
    parser.add_argument('--source_texts', type=str, default='./input_style/source.txt',
                        help='txt of style texts')
    parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
    parser.add_argument('--exp_name', type=str, default='test')

    # Training options 
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_batch_size', type=int, default=2)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch16',
                            help="CLIP model to use for the encoder")
    
    
    # Loss parameters
    parser.add_argument('--lambda_tv', type=float, default=2e-3)
    parser.add_argument('--lambda_patch', type=float, default=9000)
    parser.add_argument('--lambda_dir', type=float, default=500)
    parser.add_argument('--lambda_c', type=float, default=150)
    parser.add_argument('--n_threads', type=int, default=0)
    parser.add_argument('--thresh', type=float, default=0.7)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--num_crops', type=int, default=4)
    
    
    # StyTR basic options
    parser.add_argument('--prompt_engineering', type=bool, default=True,
                        help='whether to use prompt engineering')
    parser.add_argument('--input_size', type=int, default=224,
                        help='input image size')
    
    # StyTR Encoder options
    parser.add_argument('--encoder_embed_dim', type=int, default=512)
    parser.add_argument('--encoder_ffn_dim', type=int, default=2048)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--encoder_heads', type=int, default=8)
    parser.add_argument('--encoder_dropout', type=float, default=0.1)
    parser.add_argument('--encoder_activation', type=str, default='relu')
    parser.add_argument('--encoder_normalize_before', type=bool, default=True)
    args = parser.parse_args()
    main(args)

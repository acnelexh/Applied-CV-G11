import torch
import torch.nn.functional as F
from torch import nn
from models.transformer import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPVisionModel
from template import imagenet_templates
import math
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from models.unet import TokenDecoder

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        return x

def build_decoder(input_dimension, target_dimension):
    """
    Built decoder automatically depending on the input size
    Upsampling the input image Nx times
    """
    # assert target dimension is a multiple of input dimension
    assert(isinstance(input_dimension, int))
    assert(isinstance(target_dimension, int))
    assert(math.log2(target_dimension/input_dimension) == int(math.log2(target_dimension/input_dimension)))
    # calcualt the number of upsmampling needed
    num_upsample = int(math.log2(target_dimension/input_dimension))
    decoder = TokenDecoder(num_upsample, input_hidden_dim=512)

    dummy = torch.rand(1, 512, int(input_dimension), int(input_dimension))
    out = decoder(dummy)
    assert out.shape == (1, 3, target_dimension, target_dimension)
    return decoder

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self,
                vit_pretrain_path = "openai/clip-vit-base-patch32",
                input_size=224,
                prompt_engineering=True):

        super().__init__()

        self.mse_loss = nn.MSELoss()

        #clip stuff
        self.vision_model = CLIPVisionModel.from_pretrained(vit_pretrain_path)
        self.text_model = CLIPTextModel.from_pretrained(vit_pretrain_path)
        #self.image_processor = CLIPImageProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained(vit_pretrain_path)
        self.prompt_engineering = prompt_engineering
        self.freeze_clip()
        self.style_cache = dict() # cache the style tokens

        # build a solely encoder transformer
        # to encode vision and text
        encoder_layer = TransformerEncoderLayer(512, 8, 2048, 0.1, 'relu', True)
        self.encoder = TransformerEncoder(encoder_layer, 6)
        
        # projection layers for clip tokens
        self.fc_vision = nn.Linear(768, 512)
        self.fc_text = nn.Linear(512, 512)

        # calculate the token size
        dummy_sample = self.vision_model(torch.rand(2, 3, input_size, input_size)) 
        img_token_length = dummy_sample.last_hidden_state.shape[1] - 1
        text_token_length = 1
        # learnable positional embedding
        self.position_embedding = nn.Embedding(
            img_token_length + text_token_length, 512)
        # decoder
        self.decoder = build_decoder(int(math.sqrt(img_token_length)), input_size)

    def freeze_clip(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
    
    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
    
    def process_style(self, style):
        if not self.prompt_engineering:
            template_text = ["a photo of " + style]
        else:
            template_text = self.compose_text_with_templates(style, imagenet_templates)
        return self.tokenizer(template_text, padding=True, return_tensors="pt").to(self.text_model.device)

        
    def forward(self, content, style):
        """  
        Content: image tensor
        Style: description of style
        """
        image_tokens = self.vision_model(content)
        image_tokens = image_tokens.last_hidden_state[:, 1:, :] # get ride of cls token
        
        # cache the style tokens, recheck for gradient
        style_tokens = []
        for batch in style:
            if batch in self.style_cache:
                style_tokens.append(self.style_cache[batch])
            else:
                tmp = self.process_style(batch)
                tmp = self.text_model(**tmp)
                style_tokens.append(tmp.last_hidden_state.mean(dim=(0,1))) # average pooling
                # cache the style tokens and turn off gradient
                self.style_cache[batch] = style_tokens[-1]
                self.style_cache[batch].requires_grad = False
        style_tokens = torch.stack(style_tokens, dim=0)
        
        # project image and style tokens
        image_tokens = self.fc_vision(image_tokens)
        style_tokens = self.fc_text(style_tokens)

        # combine content and style
        input = torch.cat((image_tokens, style_tokens[:, None, :]), dim=1) 

        # add positional embedding
        input += self.position_embedding(torch.arange(input.shape[1], device=input.device))

        output_tokens = self.encoder(input)
        # decode the tokens into image
        image_tokens = output_tokens[:, :-1]
        D = int(math.sqrt(image_tokens.shape[1]))
        image_features = image_tokens.reshape(image_tokens.shape[0], D, D, 512).permute(0, 3, 1, 2)
        output = self.decoder(image_features)
        return output


def test_model():
    build_decoder(14, 224)

#test_model()

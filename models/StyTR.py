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
    
# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)



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
    
    def __init__(self, vit_pretrain_path = "openai/clip-vit-base-patch32"):

        super().__init__()

        self.mse_loss = nn.MSELoss()

        #clip stuff
        self.vision_model = CLIPVisionModel.from_pretrained(vit_pretrain_path)
        self.text_model = CLIPTextModel.from_pretrained(vit_pretrain_path)
        self.image_processor = CLIPImageProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained(vit_pretrain_path)
        self.freeze_clip()

        # build a solely encoder transformer
        # to encode vision and text
        encoder_layer = TransformerEncoderLayer(512, 8, 2048, 0.1, 'relu', True)
        self.encoder = TransformerEncoder(encoder_layer, 6)
        
        # projection layers for clip tokens
        self.fc_vision = nn.Linear(768, 512)
        self.fc_text = nn.Linear(512, 512)

        # dummy sample to figure out token size
        # calculate the input size for the model
        dummy_sample = self.image_processor(images=torch.rand(2, 3, 224, 224))
        image_size = int(dummy_sample.pixel_values[0].shape[1])

        # calculate the token size
        dummy_sample = self.vision_model(pixel_values=torch.rand(2, 3, image_size, image_size)) 
        img_token_length = dummy_sample.last_hidden_state.shape[1] - 1
        text_token_length = 1
        # learnable positional embedding
        self.position_embedding = nn.Embedding(
            img_token_length + text_token_length, 512)
        # decoder
        self.decoder = build_decoder(int(math.sqrt(img_token_length)), int(image_size))

    def freeze_clip(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
    
    def forward(self, content, style):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        if self.training == True:
            # training mode
            # input are preprocessed dictionary
            image_tokens = content['last_hidden_state']
            style_tokens = style['average_pooling']
        else:
            # eval mode
            # the input is raw image and text
            img = self.image_processor(content)
            img['pixel_values'] = torch.tensor(img['pixel_values'])
            image_tokens = self.vision_model(**img)
            image_tokens = image_tokens.last_hidden_state[:, 1:, :]

            template_text = [self.compose_text_with_templates(text, imagenet_templates) for text in style]
            style_tokens = []
            for text in template_text:
                text_tokens =  self.tokenizer(text, padding=True, return_tensors="pt")
                outputs = self.text_model(**text_tokens)
                last_hidden_state = outputs['last_hidden_state']
                #cls_token = outputs['pooler_output']
                style_tokens.append(torch.mean(last_hidden_state, dim=(0,1)))

            # project content and feats to the same dim
            image_tokens = self.fc_vision(image_tokens)
            style_tokens = self.fc_text(style_tokens)

        # project content and feats to the same dim
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

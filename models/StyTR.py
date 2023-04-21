import math
import torch
from torch import nn
import torch.nn.functional as F
from models.unet import TokenDecoder
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
from models.transformer import TransformerEncoder, TransformerEncoderLayer
from template import imagenet_templates

class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    def __init__(self, args):

        super().__init__()

        self.mse_loss = nn.MSELoss()

        #clip stuff
        self.vision_model = CLIPVisionModel.from_pretrained(args.clip_model)
        self.text_model = CLIPTextModel.from_pretrained(args.clip_model)
        #self.image_processor = CLIPImageProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained(args.clip_model)
        self.prompt_engineering = args.prompt_engineering
        self.freeze_clip()
        self.style_cache = dict() # cache the style tokens

        # build a solely encoder transformer
        # to encode vision and text
        encoder_layer = TransformerEncoderLayer(args.encoder_embed_dim,
                                                args.encoder_heads,
                                                args.encoder_ffn_dim,
                                                args.encoder_dropout,
                                                args.encoder_activation,
                                                args.encoder_normalize_before)
        self.encoder = TransformerEncoder(encoder_layer, args.encoder_depth)
        
        # projection layers for clip tokens
        self.fc_vision = nn.Linear(768, 512)
        self.fc_text = nn.Linear(512, 512)

        input_size = args.input_size
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
        
        # cache the style tokens, TODO recheck for gradient
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


# Testing ground
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

def test_model():
    build_decoder(14, 224)

#test_model()

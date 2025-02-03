import base64
from io import BytesIO
from typing import List, Union

from PIL import Image
from pydantic import BaseModel
import torch
from torchvision import models, transforms
import timm
import clip

from netdissect.imgviz import ImageVisualizer
from netdissect.imgviz import ImageVisualizer
from transformers import ViTModel, ViTImageProcessor


class Unit(BaseModel):
    model_name: str
    layer: str
    neuron_num: int


class ModelInfoWrapper:
    """Contains a reference to a model, as well as information related to that model
    """
    def __init__(self, model_name: str, device: str):
        """Loads a model, retrieves its preprocessing info, and associates 
        a device with that model

        Args:
            model_name (str) : Name of the model
            device : cuda device
        """
        self.model_name = model_name
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        # TODO - Doesn't this just get overwritten by _load_model ?
        if 'dino' in model_name or 'resnet' in model_name:
            self.preprocess = self._preprocess_imagenet
        self.model = self._load_model(model_name) #if clip, the define self.preprocess

    def preprocess_images(self, images):
        image_list = []
        if type(images) == list:
            for image in images:
                image_list.append(self.preprocess(str2image(image)).to(self.device))
            batch_tensor = torch.stack(image_list)
            return batch_tensor
        else:
            return self.preprocess(str2image(images)).unsqueeze(0).to(self.device)

    def _load_model(self, model_name: str):
        """
        Gets the model name and returns the vision model from pythorch library.
        Parameters
        ----------
        model_name : str
            The name of the model to load.
        
        Returns
        -------
        nn.Module
            The loaded PyTorch vision model.
        
        Examples
        --------
        >>> # load "resnet152"
        >>> def execute_command(model_name) -> nn.Module:
        >>>   model = load_model(model_name: str)
        >>>   return model
        """
        if model_name=='resnet152':
            resnet152 = models.resnet152(weights='IMAGENET1K_V1').to(self.device)  
            model = resnet152.eval()
        elif model_name == 'dino_vits8':
            model = ViTModel.from_pretrained('facebook/dino-vits8').to(self.device).eval()
            # self.preprocess = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
        elif model_name == "clip-RN50": 
            name = 'RN50'
            full_model, preprocess = clip.load(name)
            model = full_model.visual.to(self.device).eval()
            self.preprocess = preprocess
        elif model_name == "clip-ViT-B32": 
            name = 'ViT-B/32'
            full_model, preprocess = clip.load(name)
            model = full_model.visual.to(self.device).eval()
            self.preprocess = preprocess
        
        return model

    def _preprocess_imagenet(self, image, normalize=True, im_size=224):
        
        if normalize:
            preprocess = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
            ])
        return preprocess(image)
    
    def _replace_layers(self, model, old, new):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                self._replace_layers(module, old, new)
            if isinstance(module, old):
                ## simple module
                setattr(model, n, new())


def format_api_content(type: str, input: str):
    '''Converts input text to a format suitable for API requests
    
    Parameters
    ----------
    type : str
        The type of content to be converted. Can be "text" or "image".
    input : strself._self._
        The input content to be converted.'''

    if input is None:
        return {"type": "text", "text": "None"}
    elif type == "text":
        return {"type": "text", "text": str(input)}
    elif type == "image_url":
        return {"type": "image_url", "image_url": {"url": base64_to_url(input)}}
    else:
        raise ValueError(f"Unsupported type: {type}")


def is_base64(s: str)->bool:
    """
    Check if a string is in valid Base64 format.

    Parameters:
    -----------
    s : str
        The string to check.

    Returns:
    --------
    bool
        True if the string is in valid Base64 format, False otherwise.
    
    Notes:
    ------
    Non-zero chance that the model will return a base64 string that is not a
    valid image. However, since spaces aren't allowed, the model would need to
    return a one word response.
    """
    # Try to decode the string
    try:
        decoded = base64.b64decode(s)
        # Check if the decoded string can be encoded back to the original
        return base64.b64encode(decoded).decode() == s
    except Exception:
        return False


def base64_to_url(base64_str: str):
    '''Converts a base64 string to a URL
    
    Parameters
    ----------
    base64_str : str
        The base64 string to be converted.'''
    
    return "data:image/jpeg;base64," + base64_str

def generate_masked_image(image:torch.Tensor, mask:torch.Tensor, threshold:float):
    vis = ImageVisualizer(224, image_size=224, source='imagenet')
    masked_tensor = vis.pytorch_masked_image(image, activations=mask, unit=None, level=threshold, outside_bright=0.25)
    masked_image = Image.fromarray(masked_tensor.permute(1, 2, 0).byte().cpu().numpy())
    masked_image_str = image2str(masked_image)
    return(masked_image_str)


def image2str(image)->str:
    if type(image) == str:
        return image
    if isinstance(image, list):
        return [image2str(img) for img in image]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('ascii')
    return(image)


def str2image(image_str):
    if type(image_str) == Image.Image:
        return image_str
    if isinstance(image_str, list):
        return [str2image(img) for img in image_str]

    # Converts a Base64 encoded string to an image.
    img_bytes = base64.b64decode(image_str)
    img_buffer = BytesIO(img_bytes)
    img = Image.open(img_buffer)
    return img


def resize_image(image: Image.Image, size=(256, 256)) -> Image.Image:
    return image.resize(size)

def merge_images_horizontally(images: List[Image.Image], gap: int = 10) -> Image.Image:
    imgs = [resize_image(image) for image in images]
    total_width = sum(img.width for img in imgs) + gap * (len(imgs) - 1)
    height = imgs[0].height

    merged = Image.new("RGB", (total_width, height))

    x_offset = 0
    for img in imgs:
        merged.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    return merged


def merge_images_vertically(images: List[Image.Image], gap: int = 10) -> Image.Image:
    imgs = images
    total_height = sum(img.height for img in imgs) + gap * (len(imgs) - 1)
    width = max(img.width for img in imgs)

    merged = Image.new("RGB", (width, total_height))

    y_offset = 0
    for img in imgs:
        merged.paste(img, (0, y_offset))
        y_offset += img.height + gap

    return merged


def generate_gridded_image(images: Union[List[Image.Image], List[str]]):
    n_images = len(images)

    # If images stored in base64, convert to PIL Image objects
    images = [str2image(image) for image in images]
    
    # Merge images from the same dataset horizontally
    merged_images_first = merge_images_horizontally(
        images[: n_images // 2]
    )
    merged_images_second = merge_images_horizontally(
        images[n_images // 2 :]
    )
    
    # Merge the resulting images from different datasets vertically
    final_merged_image = merge_images_vertically(
        [
            merged_images_first,
            merged_images_second
        ]
    )

    return final_merged_image
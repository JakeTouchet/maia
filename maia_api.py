# Standard library imports
import math
import os,sys
import time
from io import BytesIO
from typing import Dict, List, Tuple, Union, Iterable

# Third-party imports
import openai
import torch
import torch.nn.functional as F
from baukit import Trace
from diffusers import (
    AutoPipelineForText2Image,
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline
)
from PIL import Image
import numpy as np

# Local imports
from utils.call_agent import ask_agent
from utils.api_utils import is_base64, format_api_content, generate_masked_image, image2str, str2image, Unit, ModelInfoWrapper
from utils.DatasetExemplars import DatasetExemplars
from utils.main_utils import generate_numbered_path
from utils.InterpAgent import InterpAgent
from synthetic_neurons_dataset import synthetic_neurons


class System:
    """
    A Python class for interfacing with specified units within vision models.
    
    Attributes
    ----------
    units : List[Unit]
        A list of units, each containing the model name, layer name, and neuron number.
    unit : Unit
        The current unit being analyzed.
    model_wrapper : ModelInfoWrapper
        The model wrapper for the current unit.
    model_dict : Dict[str, ModelInfoWrapper]
        A dictionary of model names and their corresponding ModelInfoWrapper objects.
    device : torch.device
        The device (CPU/GPU) used for computations.
    threshold : int
        The current activation threshold for neuron analysis.
    thresholds : Dict
        A dictionary containing the threshold values for each unit.

    Methods
    -------
    call_units(self, image_list: List[torch.Tensor], unit_ids:List[int])->List[List[Tuple[float, str]]]]]
        For each image, returns each unit's maximum activation value
        (in int format) over that image. Also returns masked images that
        highlight the regions of the image where the activations are highest
        (encoded into a Base64 string).

    """
    def __init__(self, model_name: str, layer: str, neuron_num: int, thresholds: Dict, device: Union[int, str]):
        """
        Initializes a system for interfacing with a set of specified units.
        Parameters
        -------
        unit_dict : dict
            {
            model_name: {
                    layer_name: neuron_list
                    }
            }
        thesholds : dict
            Contains the threshold values for each unit
        device : str
            The computational device ('cpu' or 'cuda').
        """
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu") 
        # Loads and stores preprocessing info for the model being experimented with
        self.model_wrapper = ModelInfoWrapper(model_name, device)
        # Select first unit as current unit
        self.unit = Unit(model_name=model_name, layer=layer, neuron_num=neuron_num)

        if thresholds:
            self.threshold = thresholds[self.unit.model_name][self.unit.layer][self.unit.neuron_num]
        else:
            self.threshold = 0

    def call_neuron(self, image_list: List[torch.Tensor])->Tuple[List[float], List[str]]:
        """
        For each specified unit, returns the unit’s maximum activation value
        (in int format) for each image. Also returns masked images that
        highlight the regions of the image where the activations are highest
        (encoded into a Base64 string).
        
        Parameters
        ----------
        image_list : List[torch.Tensor]
            The input image
        unit_ids : List[int]
            The unit ids to retrieve activations for.
        
        Returns
        -------
        List[List[Tuple[float, str]]]
            For each unit, stores the maximum activations and masked images as a tuple.
            If only one unit is specified, returns a single tuple.
        
        Examples
        --------
        >>> # Test the activation value of a single unit for a prompt
        >>> prompt = ["a man eating a gargantuan sandwich"]
        >>> images = tools.text2image(prompt)
        >>> 
        >>> tools.display(*activations, *masked_image)
        >>>
        >>> # Test the activation value of multiple units for the prompt "a dog standing on the grass"
        >>> prompt = ["a dog standing on the grass"]
        >>> images = tools.text2image(prompt)
        >>> unit_ids = [0, 1]  # Example unit IDs to test
        >>> unit_data = system.call_units(images, unit_ids)
        >>> for activations, masked_images in unit_data:
        >>>    tools.display(*masked_images, *activations)
        >>>
        >>> # Test the activation value of multiple units for multiple prompts
        >>> prompt_list = ["a fox and a rabbit watch a movie under a starry night sky",
        >>>                "a fox and a bear watch a movie under a starry night sky",
        >>>                "a fox and a rabbit watch a movie at sunrise"]
        >>> images = tools.text2image(prompt_list)
        >>> unit_ids = [0, 1]  # Example unit IDs to test
        >>> unit_data = system.call_units(images, unit_ids)
        >>> for i in range(len(images)):
        >>>     tools.display(prompt_list[i], images[i])
        >>>     for j in range(len(unit_data)):
        >>>         activation, masked_image = unit_data[j][i]
        >>>         tools.display(f"unit {j}")
        >>>         tools.display(masked_image, activation)
        """

        activations, masked_images = [], []
        for image in image_list:
            if  image==None: #for dalle
                activation = None
                masked_image = None
            else:
                if self.unit.layer == 'last':
                    image = self.model_wrapper.preprocess_images(image)
                    acts, image = self._calc_class(image)    
                    activation = acts
                    masked_image = None
                else:
                    image = self.model_wrapper.preprocess_images(image)
                    acts,masks = self._calc_activations(image)    
                    ind = torch.argmax(acts).item()
                    activation = acts[ind].item()
                    masked_image = generate_masked_image(image[ind], masks[ind], self.threshold)
            activations.append(activation)
            masked_images.append(masked_image)

        return activations, masked_images

    @staticmethod
    def _spatialize_vit_mlp(hiddens: torch.Tensor) -> torch.Tensor:
        """Make ViT MLP activations look like convolutional activations.
    
        Each activation corresponds to an image patch, so we can arrange them
        spatially. This allows us to use all the same dissection tools we
        used for CNNs.
    
        Args:
            hiddens: The hidden activations. Should have shape
                (batch_size, n_patches, n_units).
    
        Returns:
            Spatially arranged activations, with shape
                (batch_size, n_units, sqrt(n_patches - 1), sqrt(n_patches - 1)).
        """
        batch_size, n_patches, n_units = hiddens.shape

        # Exclude CLS token.
        hiddens = hiddens[:, 1:]
        n_patches -= 1

        # Compute spatial size.
        size = math.isqrt(n_patches)
        assert size**2 == n_patches

        # Finally, reshape.
        return hiddens.permute(0, 2, 1).reshape(batch_size, n_units, size, size)

    def _calc_activations(self, image: torch.Tensor)->Tuple[int, torch.Tensor]:
        """"
        Returns the neuron activation for the input image, as well as the activation map of the neuron over the image
        that highlights the regions of the image where the activations are higher (encoded into a Base64 string).
    
        Parameters
        ----------
        image : torch.Tensor
            The input image in PIL format.
        
        Returns
        -------
        Tuple[int, torch.Tensor]
            Returns the maximum activation value of the neuron on the input image and a mask
        """
        with Trace(self.model_wrapper.model, self.unit.layer) as ret:
            _ = self.model_wrapper.model(image)
            hiddens = ret.output

        if "dino" in self.model_wrapper.model_name:
            hiddens = self._spatialize_vit_mlp(hiddens)

        batch_size, channels, *_ = hiddens.shape
        activations = hiddens.permute(0, 2, 3, 1).reshape(-1, channels)
        pooled, _ = hiddens.view(batch_size, channels, -1).max(dim=2)
        neuron_activation_map = hiddens[:, self.unit.neuron_num, :, :]
        return(pooled[:,self.unit.neuron_num], neuron_activation_map)

    def _calc_class(self, image: torch.Tensor)->Tuple[int, torch.Tensor]:
        """"
        Returns the neuron activation for the input image, as well as the activation map of the neuron over the image
        that highlights the regions of the image where the activations are higher (encoded into a Base64 string).
    
        Parameters
        ----------
        image : torch.Tensor
            The input image in PIL format.
        
        Returns
        -------
        Tuple[int, torch.Tensor]
            Returns the maximum activation value of the neuron on the input image and a mask
        """
        logits = self.model_wrapper.model(image)
        prob = F.softmax(logits, dim=1)
        image_calss = torch.argmax(logits[0])
        activation = prob[0][image_calss]
        activation = activation.item()
        activation = round(activation, 4)
        return activation.item(), image

class SyntheticSystem:
    """
    A Python class containing the vision model and the specific neuron to interact with.
    
    Attributes
    ----------
    neuron_num : int
        The serial number of the neuron.
    layer : string
        The name of the layer where the neuron is located.
    model_name : string
        The name of the vision model.
    model : nn.Module
        The loaded PyTorch model.
    neuron : callable
        A lambda function to compute neuron activation and activation map per input image. 
        Use this function to test the neuron activation for a specific image.
    device : torch.device
        The device (CPU/GPU) used for computations.

    Methods
    -------
    load_model(model_name: str)->nn.Module
        Gets the model name and returns the vision model from PyTorch library.
    call_neuron(image_list: List[torch.Tensor])->Tuple[List[int], List[str]]
        returns the neuron activation for each image in the input image_list as well as the activation map 
        of the neuron over that image, that highlights the regions of the image where the activations 
        are higher (encoded into a Base64 string).
    """
    def __init__(self, neuron_num: int, neuron_labels: str, neuron_mode: str, device: str):
        
        self.neuron_labels = neuron_labels
        self.neuron = synthetic_neurons.SAMNeuron(neuron_labels, neuron_mode)
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")       
        self.threshold = 0

        self.unit = Unit(model_name="synthetic", layer=neuron_mode, neuron_num=neuron_num)


    def call_neuron(self, image_list: List[torch.Tensor])->Tuple[List[int], List[str]]:
        """
        The function returns the neuron’s maximum activation value (in int format) over each of the images in the list as well as the activation map of the neuron over each of the images that highlights the regions of the image where the activations are higher (encoded into a Base64 string).
        
        Parameters
        ----------
        image_list : List[torch.Tensor]
            The input image
        
        Returns
        -------
        Tuple[List[int], List[str]]
            For each image in image_list returns the maximum activation value of the neuron on that image, and a masked images, 
            with the region of the image that caused the high activation values highlighted (and the rest of the image is darkened). Each image is encoded into a Base64 string.

        
        Examples
        --------
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass"
        >>> def execute_command(system, prompt_list) -> Tuple[int, str]:
        >>>     prompt = ["a dog standing on the grass"]
        >>>     image = text2image(prompt)
        >>>     activation_list, activation_map_list = system.call_neuron(image)
        >>>     return activation_list, activation_map_list
        >>> # test the activation value of the neuron for the prompt “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise”
        >>> def execute_command(system.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [[“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]]
        >>>     images = text2image(prompt_list)
        >>>     activation_list, activation_map_list = system.call_neuron(images)
        >>>     return activation_list, activation_map_list
        """
        activation_list = []
        masked_images_list = []
        for images in image_list:
            if images==None: #for dalle
                activation_list.append([])
                masked_images_list.append([])
            else:
                if type(images) == list:
                    images = [str2image(image) for image in images]
                else:
                    images = [str2image(images)]
                acts, _, _, masked_images = self.neuron.calc_activations(images)
                masks_str = []
                for masked_image in masked_images:
                    masked_image = image2str(masked_image)
                    masks_str.append(masked_image)
                activation_list.extend(acts)
                masked_images_list.extend(masks_str)

        return activation_list,masked_images_list

class Tools:
    """
    A Python class containing tools to interact with the units implemented in the system class, 
    in order to run experiments on it.
    
    Attributes
    ----------
    text2image_model_name : str
        The name of the text-to-image model.
    text2image_model : any
        The loaded text-to-image model.
    images_per_prompt : int
        Number of images to generate per prompt.
    path2save : str
        Path for saving output images.
    threshold : any
        Activation threshold for neuron analysis.
    device : torch.device
        The device (CPU/GPU) used for computations.
    experiment_log: str
        A log of all the experiments, including the code and the output from the neuron
        analysis.
    exemplars : Dict
        A dictionary containing the exemplar images for each unit.
    exemplars_activations : Dict
        A dictionary containing the activations for each exemplar image.
    exemplars_thresholds : Dict
        A dictionary containing the threshold values for each unit.
    results_list : List
        A list of the results from the neuron analysis.


    Methods
    -------
    dataset_exemplars(self, unit_ids: List[int], system: System)->List[List[Tuple[float, str]]]:
        Retrieves the activations and exemplar images for a list of units.
    edit_images(self, image_prompts : List[Image.Image], editing_prompts : List[str]):
        Generate images from a list of prompts, then edits each image with the
        corresponding editing prompt.
    text2image(self, prompt_list: List[str]) -> List[torch.Tensor]:
        Gets a list of text prompt as an input, generates an image for each prompt in the list using a text to image model.
        The function returns a list of images.
    summarize_images(self, image_list: List[str]) -> str:
        Gets a list of images and describes what is common to all of them, focusing specifically on unmasked regions.
    sampler(act: any, imgs: List[any], mask: any, prompt: str, method: str = 'max') -> Tuple[List[int], List[str]]
        Processes images based on neuron activations.
    describe_images(self, image_list: List[str], image_title:List[str]) -> str:
        Generates descriptions for a list of images, focusing specifically on highlighted regions.
    display(self, *args: Union[str, Image.Image]):
        Displays a series of images and/or text in the chat, similar to a Jupyter notebook.

    """

    def __init__(
                self, 
                path2save: str, 
                device: str,  
                agent: InterpAgent,
                system: System,
                DatasetExemplars: DatasetExemplars = None, 
                images_per_prompt=10, 
                text2image_model_name='sd', 
                image2text_model_name='gpt-4o'):
        """
        Initializes the Tools object.

        Parameters
        ----------
        path2save : str
            Path for saving output images.
        device : str
            The computational device ('cpu' or 'cuda').
        DatasetExemplars : object
            an object from the class DatasetExemplars
        images_per_prompt : int
            Number of images to generate per prompt.
        text2image_model_name : str
            The name of the text-to-image model.
        """
        self.image2text_model_name = image2text_model_name
        self.text2image_model_name = text2image_model_name
        self.agent = agent
        self.system = system
        self.images_per_prompt = images_per_prompt
        self.p2p_model_name = 'ip2p'
        self.path2save = path2save
        self.im_size = 224
        self.DatasetExemplars = DatasetExemplars
        self.activation_threshold = 0
        self.results_list = []

        if DatasetExemplars is not None:
            self.exemplars = DatasetExemplars.exemplars
            self.exemplars_activations = DatasetExemplars.activations
            self.exempalrs_thresholds = DatasetExemplars.thresholds
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.text2image_model = self._load_text2image_model(model_name=text2image_model_name)
        self.p2p_model = self._load_pix2pix_model(model_name=self.p2p_model_name) # consider maybe adding options for other models like pix2pix zero
        self.html_path = generate_numbered_path(os.path.join(path2save, "experiment"), ".html")

    def dataset_exemplars(self)->Tuple[List[float], List[str]]:
        """
        Retrieves the activations and exemplar images the specified units.

        Parameters
        ----------
        unit_ids : Union[List[int], int]
            Unit ids to retrieve exemplars for.

        Returns
        -------
        List[List[Tuple[float, str]]]
            For each unit, stores the maximum activations and masked images as a tuple.

        Example
        -------
        >>> # Display the exemplars and activations for a list of units
        >>> unit_ids = [0, 1]
        >>> exemplar_data = tools.dataset_exemplars(unit_ids, system)
        >>> for i in range(len(exemplar_data)):
        >>>     tools.display(f"unit {unit_ids[i]}: ")
        >>>     for activation, masked_image in exemplar_data[i]:
        >>>         tools.display(masked_image, activation)
        >>> # Test each unit on the other's exemplars
        >>> unit_ids = [0, 1]
        >>> exemplar_data = tools.dataset_exemplars(unit_ids, system, mask=False)
        >>> _, exemplars_0 = zip(*exemplar_data[0])
        >>> _, exemplars_1 = zip(*exemplar_data[1])

        >>> unit_data = []
        >>> unit_data.append(system.call_units(exemplars_1, [0])[0])
        >>> unit_data.append(system.call_units(exemplars_0, [1])[0])
        >>> for i in range(len(unit_data)):
        >>>     tools.display(f"unit {i}: ")
        >>>     unit_datum = unit_data[0]
        >>>     for activation, masked_image in unit_datum:
        >>>         tools.display(masked_image, activation)
        """
        unit = self.system.unit
        image_list = self.exemplars[unit.model_name][unit.layer][unit.neuron_num]
        activation_list = self.exemplars_activations[unit.model_name][unit.layer][unit.neuron_num]
        activation_list = [round(activation, 2) for activation in activation_list]

        return activation_list, image_list
    
    # TODO - Rework, look at editing tools
    def edit_images(self, images: Union[List[Image.Image], List[str]], prompts: List[str]):
        """
        Generate images from a list of prompts, then edits each image with the
        corresponding editing prompt. Important note: Do not use negative
        terminology such as "remove ...", try to use terminology like "replace
        ... with ..." or "change the color of ... to" The function returns a
        list of images and list of the relevant prompts.

        Parameters
        ----------
        image_prompts : List[str]
            A list of input ptompts to generate images according to, these
            images are to be edited by the prompts in editing_prompts.
        editing_prompts : List[str]
            A list of instructions for how to edit the images in image_list.
            Should be the same length as image_list.

        Returns
        -------
        List[Image.Image], List[str]
            A list of images and a list of all the prompts that
            were used in the experiment, in the same order as the images
        """

        # Generate list of edited images from editing instructions
        # TODO - Assumes one image per prompt, talk to Tamar
        images = [image[0] for image in images]
        edited_images = self.p2p_model(prompts, images).images

        return edited_images

    def text2image(self, prompt_list: List[str]) -> List[torch.Tensor]:
        """Gets a list of text prompt as an input, generates an image for each prompt in the list using a text to image model.
        The function returns a list of images.

        Parameters
        ----------
        prompt_list : List[str]
            A list of text prompts for image generation.

        Returns
        -------
        List[Image.Image]
            A list of images, corresponding to each of the input prompts. 

        """
        image_list = [] 
        for prompt in prompt_list:
            images = self._prompt2image(prompt)
            # TODO - Concatenates, talk to Tamar
            image_list.append(images)
        return image_list


    def summarize_images(self, image_list: List[str]) -> str:
        """
        Gets a list of images and describes what is common to all of them, focusing specifically on unmasked regions.


        Parameters
        ----------
        image_list : list
            A list of images in Base64 encoded string format.
        
        Returns
        -------
        str
            A string with a descriptions of what is common to all the images.

        Example
        -------
        >>> # Summarize a unit's dataset exemplars
        >>> _, exemplars = tools.dataset_exemplars([0], system)[0] # Get exemplars for unit 0
        >>> summarization = tools.summarize_images(image_list)
        >>> tools.display("Unit 0 summarization: ", summarization)
        >>> 
        >>> # Summarize what's common amongst two sets of exemplars
        >>> exemplars_data = tools.dataset_exemplars([0,1], system)
        >>> all_exemplars = []
        >>> for _, exemplars in exemplars_data:
        >>>     all_exemplars += exemplars
        >>> summarization = tools.summarize_images(all_exemplars)
        >>> tools.display("All exemplars summarization: ", summarization)
        """
        instructions = '''
        What do all the unmasked regions of these images have in common? There
        might be more then one common concept, or a few groups of images with
        different common concept each. In these cases return all of the
        concepts.. 
        Return your description in the following format: 

        [COMMON]:
        <your description>.
        '''
        history = [{'role':'system', 'content':'You are a helpful assistant who views/compares partially or fully masked images.'}]
        user_content = [{"type":"text", "text": instructions}]
        for ind,image in enumerate(image_list):
            user_content.append(format_api_content("image_url", image))
        history.append({'role': 'user', 'content': user_content})
        description = ask_agent(self.image2text_model_name,history)
        if isinstance(description, Exception): return description
        return description

    def sampler(self, act, imgs, mask, prompt, threshold, method = 'max'):  
        if method=='max':
            max_ind = torch.argmax(act).item()
            masked_image_max = self.generate_masked_image(image=imgs[max_ind],mask=mask[max_ind],path2save=f"{self.path2save}/{prompt}_masked_max.png", threshold=threshold)
            acts= max(act).item()
            ims = masked_image_max
        elif method=='min_max':
            max_ind = torch.argmax(act).item()
            min_ind = torch.argmin(act).item()
            masked_image_max = self.generate_masked_image(image=imgs[max_ind],mask=mask[max_ind],path2save=f"{self.path2save}/{prompt}_masked_max.png", threshold=threshold)
            masked_image_min = self.generate_masked_image(image=imgs[min_ind],mask=mask[min_ind],path2save=f"{self.path2save}/{prompt}_masked_min.png", threshold=threshold)
            acts = []
            ims = [] 
            acts.append(max(act).item())
            acts.append(min(act).item())
            ims.append(masked_image_max)
            ims.append(masked_image_min)
        return acts, ims

    def describe_images(self, image_list: List[str], image_title:List[str]) -> str:
        """
        Provides impartial description of the highlighted image regions within an image.
        Generates textual descriptions for a list of images, focusing specifically on highlighted regions.
        This function translates the visual content of the highlighted region in the image to a text description. 
        The function operates independently of the current hypothesis list and thus offers an impartial description of the visual content.        
        It iterates through a list of images, requesting a description for the 
        highlighted (unmasked) regions in each synthetic image. The final descriptions are concatenated 
        and returned as a single string, with each description associated with the corresponding 
        image title.

        Parameters
        ----------
        image_list : List[str]
            A list of images in Base64 encoded string format.
        image_title : List[str]
            A list of titles for each image in the image_list.

        Returns
        -------
        str
            A concatenated string of descriptions for each image, where each description 
            is associated with the image's title and focuses on the highlighted regions 
            in the image.

        Example
        -------
        >>> prompt_list = [“a man smiling”, 
                            “a man frowning”,
                            “a man talking”]
        >>> images = tools.text2image(prompt_list)
        >>> activation_list, masked_images = system.units(images, [0])[0]
        >>> descriptions = tools.describe_images(masked_images, prompt_list)
        >>> tools.display(*descriptions)
        """
        description_list = ''
        instructions = "Do not describe the full image. Please describe ONLY the unmasked regions in this image (e.g. the regions that are not darkened). Be as concise as possible. Return your description in the following format: [highlighted regions]: <your concise description>"
        time.sleep(60)
        for ind, image in enumerate(image_list):
            history = [{'role':'system', 
                        'content':'you are an helpful assistant'},
                        {'role': 'user', 
                         'content': 
                         [format_api_content("text", instructions),
                           format_api_content("image_url", image)]}]
            description = ask_agent(self.image2text_model_name, history)
            if isinstance(description, Exception): return description_list
            description = description.split("[highlighted regions]:")[-1]
            description = " ".join([f'"{image_title[ind]}", highlighted regions:',description])
            description_list += description + '\n'
        return description_list

    def display(self, *args: Union[str, Image.Image, Iterable]):
        """
        Displays a series of images and/or text in the chat, similar to a Jupyter notebook.
        Recursively displays nested iterables.
        
        Parameters
        ----------
        *args : Union[str, Image.Image, Iterable]
            The content to be displayed in the chat. Can be multiple strings or Image objects.

        Notes
        -------
        Displays directly to chat interface.

        Example
        -------
        >>> # Display a single image
        >>> prompt = ["a dog standing on the grass"]
        >>> images = tools.text2image(prompt)
        >>> tools.display(*images)
        >>>
        >>> # Display a list of images
        >>> prompt_list = ["A green creature",
        >>>                 "A red creature",
        >>>                 "A blue creature"]
        >>> images = tools.text2image(prompt_list)
        >>> tools.display(*images)
        """
        output = []
        for item in args:
            # Check if tuple or list
            if isinstance(item, (list, tuple)):
                output.extend(self._display_helper(*item))
            else:
                output.append(self._process_chat_input(item))
        self.agent.update_experiment_log(role='user', content=output)

    def _display_helper(self, *args: Union[str, Image.Image]):
        '''Helper function for display to recursively handle iterable arguments.'''
        output = []
        for item in args:
            if isinstance(item, (list, tuple)):
                output.extend(self._display_helper(*item))
            else:
                output.append(self._process_chat_input(item))
        return output

    def _process_chat_input(self, content: Union[str, Image.Image]) -> Dict[str, str]:
        '''Processes the input content for the chatbot.
        
        Parameters
        ----------
        content : Union[str, Image.Image]
            The input content to be processed.'''

        if is_base64(content):
            return format_api_content("image_url", content)
        elif isinstance(content, Image.Image):
            return format_api_content("image_url", image2str(content))
        else:
            return format_api_content("text", content)

    # TODO - Try stabilityai/stable-diffusion-xl-refiner-1.0
    def _load_pix2pix_model(self, model_name):
        """
        Loads a pix2pix image editing model.

        Parameters
        ----------
        model_name : str
            The name of the pix2pix model.

        Returns
        -------
        The loaded pix2pix model.
        """
        if model_name == "ip2p": # instruction tuned pix2pix model
            device = self.device
            model_id = "timbrooks/instruct-pix2pix"
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
            pipe = pipe.to(device)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

            # Set progress bar to quiet mode
            pipe.set_progress_bar_config(disable=True)
            return pipe
        else:
            raise("unrecognized pix2pix model name")


    def _load_text2image_model(self,model_name):
        """
        Loads a text-to-image model.

        Parameter
        ----------
        model_name : str
            The name of the text-to-image model.

        Returns
        -------
        The loaded text-to-image model.
        """
        if model_name == "sd":
            device = self.device
            model_id = "runwayml/stable-diffusion-v1-5"
            sdpipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            sdpipe = sdpipe.to(device)

            # Set progress bar to quiet mode to not clog error output.
            sdpipe.set_progress_bar_config(disable=True)

            return sdpipe
        elif model_name == "sdxl-turbo":
            device = self.device
            model_id = "stabilityai/sdxl-turbo"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            pipe = pipe.to(device)
            return pipe
        elif model_name == "dalle":
            pipe = None
            return pipe
        else:
            raise("unrecognized text to image model name")

    def _prompt2image(self, prompt, images_per_prompt=None):
        if images_per_prompt == None: 
            images_per_prompt = self.images_per_prompt
        if self.text2image_model_name == "sd":
            prompts = [prompt] * images_per_prompt
            images = []
            for prompt in prompts:
                images.append(self.text2image_model(prompt).images[0])
        elif self.text2image_model_name == "dalle":
            if images_per_prompt > 1:
                raise("cannot use DALLE with 'images_per_prompt'>1 due to rate limits")
            else:
                try:
                    prompt = "a photo-realistic image of " + prompt
                    response = openai.Image.create(prompt=prompt, 
                                                   model="dall-e-3",
                                                   n=1, 
                                                   size="1024x1024",
                                                   quality="hd",
                                                   response_format="b64_json"
                                                   )
                    image = response.data[0].b64_json
                    images = [str2image(image)]
                except Exception as e:
                    raise(e)
        images = [im.resize((self.im_size, self.im_size)) for im in images]
        return images

    # TODO - Make experiment log more human friendly to view
    def generate_html(self, experiment_log: List[Dict], name="experiment", line_length=100):
        # Generates an HTML file with the experiment log.
        html_string = f'''<html>
        <head>
        <title>Experiment Log</title>
        <!-- Include Prism Core CSS (Choose the theme you prefer) -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet" />
        <!-- Include Prism Core JavaScript -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/prism.min.js"></script>
        <!-- Include the Python language component for Prism -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-python.min.js"></script>
        </head> 
        <body>
        <h1>{self.path2save}</h1>'''

        # don't plot system+user prompts (uncomment if you want the html to include the system+user prompts)
        '''
        html_string += f"<h2>{experiment_log[0]['role']}</h2>"
        html_string += f"<pre><code>{experiment_log[0]['content']}</code></pre><br>"
        
        html_string += f"<h2>{experiment_log[1]['role']}</h2>"
        html_string += f"<pre>{experiment_log[1]['content'][0]}</pre><br>"
        initial_images = ''
        initial_activations = ''
        for cont in experiment_log[1]['content'][1:]:
            if isinstance(cont, dict):
                initial_images += f"<img src="data:image/png;base64,{cont['image']}"/>"
            else:
                initial_activations += f"{cont}    "
        html_string +=  initial_images
        html_string += f"<p>Activations:</p>"
        html_string += initial_activations
        '''

        for entry in experiment_log:      
            if entry['role'] == 'assistant':
                html_string += f"<h2>MAIA</h2>"  
                text = entry['content'][0]['text']

                html_string += f"<pre>{text}</pre><br>"
                html_string += f"<h2>Experiment Execution</h2>"  
            else:
                for content_entry in entry['content']:

                    if "image_url" in content_entry["type"]:
                        html_string += f'''<img src="{content_entry['image_url']['url']}"/>'''  
                    elif "text" in content_entry["type"]:
                        html_string += f"<pre>{content_entry['text']}</pre>"
        html_string += '</body></html>'

        # Save
        with open(self.html_path, "w") as file:
            file.write(html_string)
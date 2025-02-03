import os
import numpy as np
import base64
from PIL import Image
from collections import defaultdict
from IPython import embed
from typing import Dict, List


class DatasetExemplars():
    """
    A class for performing network dissection on a given neural network model.

    This class analyzes specific layers and units of a neural network model to 
    understand what each unit in a layer has learned. It uses a set of exemplar 
    images to activate the units and stores the resulting activations, along with
    visualizations of the activated regions in the images.

    Attributes
    ----------
    path2exemplars : str
        Path to the directory containing the exemplar images.
    n_exemplars : int
        Number of exemplar images to use for each unit.
    path2save : str
        Path to the directory where the results will be saved.
    unit_dict : dict
        Dict containing path to each unit: model_name->layer_name->neuron_num
        {
        model_name: {
                layer_name: neuron_list
                }
        }
    im_size : int, optional
        Size to which the images will be resized (default is 224).
    all_units : bool, optional
        Dissect all units in each layer. False by default.

    Methods
    -------
    net_dissect(layer: str, im_size: int=224)
        Dissects the specified layer of the neural network, analyzing the response
        to the exemplar images and saving visualizations of activated regions.
    """

    def __init__(self, path2exemplars: str, n_exemplars: int, path2save: str, unit_dict: Dict[str, Dict[str, List[int]]], im_size=224, all_units=False):

        """
        Constructs all the necessary attributes for the DatasetExemplars object.

        Parameters
        ----------
        path2exemplars : str
            Path to the directory containing the exemplar images.
        n_exemplars : int
            Number of exemplar images to use for each unit.
        path2save : str
            Path to the directory where the results will be saved.
        unit_dict : dict
            Dict containing path to each unit: model_name->layer_name->neuron_num
            {
            model_name: {
                    layer_name: neuron_list
                    }
            }
        im_size : int, optional
            Size to which the images will be resized (default is 224).
        all_units : bool, optional
            Dissect all units in each layer. False by default.
        """

        self.path2exemplars = path2exemplars
        self.n_exemplars = n_exemplars
        self.path2save = path2save
        self.unit_dict = unit_dict
        self.im_size = im_size
        self.exemplars = defaultdict(lambda : defaultdict(dict))
        self.activations = defaultdict(lambda : defaultdict(dict))
        self.thresholds = defaultdict(lambda : defaultdict(dict))
        for model_name, layers in unit_dict.items():
            if model_name == "synthetic":
                continue
            for layer in layers:
                if not all_units:
                    exemplars, activations, thresholds = self.net_dissect(model_name, layer, unit_dict[model_name][layer])
                else:
                    exemplars, activations, thresholds = self.net_dissect(model_name, layer, None)
                self.exemplars[model_name][layer] = exemplars
                self.activations[model_name][layer] = activations
                self.thresholds[model_name][layer] = thresholds
    
    def get_masked_exemplars_path(self, model_name: str, layer: str, unit: int):
        return os.path.join(self.path2save,'dataset_exemplars',model_name,layer,str(unit),'netdisect_exemplars')

    def net_dissect(self, model_name, layer, units, im_size=224):
        """
        Dissects the specified layer of the neural network.
        This method analyzes the response of units in the specified layer to
        the exemplar images. It generates and saves visualizations of both
        the original and masked activated regions in these images.
        
        Parameters
        ----------
        model_name : str
            The name of the model containing the layer to be dissected
        layer : str
            The name of the layer to be dissected.
        units : list or None
            List of specific units to analyze, or None to analyze all units.
        im_size : int, optional
            Size to which the images will be resized for visualization 
            (default is 224).
        
        Returns
        -------
        tuple
            A tuple containing lists of unmasked images, masked images, 
            activations, and thresholds for the specified layer. 
            The images are Base64 encoded strings.
        """
        exp_path = os.path.join(self.path2exemplars, model_name, 'imagenet', layer)
        activations = np.loadtxt(os.path.join(exp_path, 'activations.csv'), delimiter=',')  # units * exemplars
        thresholds = np.loadtxt(os.path.join(exp_path, 'thresholds.csv'), delimiter=',')  # units
        image_array = np.load(os.path.join(exp_path, 'images.npy')) # units * exemplars * 3 * 224 * 224 (needs to be flattened for 'last' layer)
        mask_array = np.load(os.path.join(exp_path, 'masks.npy')) # units * exemplars * 224 * 224 (ditto)
        if layer == "last":
            # Squeeze out singletons
            image_array = np.squeeze(image_array)
        
        all_masked_images = []
        
        # Loop through each unit, determined by index in the activations csv
        for unit in range(activations.shape[0]):
            curr_masked_list = []
            
            if units is not None and unit not in units:
                all_masked_images.append(curr_masked_list)
                continue
            
            for exemplar_idx in range(min(activations.shape[1], self.n_exemplars)):
                save_path = self.get_masked_exemplars_path(model_name, layer, unit)
                os.makedirs(save_path, exist_ok=True)
                
                # Process masked image
                masked_path = os.path.join(save_path, f'{exemplar_idx}_masked.png')
                if os.path.exists(masked_path):
                    with open(masked_path, "rb") as image_file:
                        masked_b64 = base64.b64encode(image_file.read()).decode('utf-8')
                else:
                    curr_image = image_array[unit, exemplar_idx]
                    if layer != "last":
                        curr_mask = np.repeat(mask_array[unit, exemplar_idx], 3, axis=0)
                        inside = np.array(curr_mask > 0)
                        outside = np.array(curr_mask == 0)
                        masked_image = curr_image * inside + 0 * curr_image * outside
                    else:
                        # No mask needed for last layer
                        masked_image = curr_image
                    masked_image = Image.fromarray(np.transpose(masked_image, (1, 2, 0)).astype(np.uint8))
                    masked_image = masked_image.resize([self.im_size, self.im_size], Image.Resampling.LANCZOS)
                    masked_image.save(masked_path, format='PNG')
                    
                    with open(masked_path, "rb") as image_file:
                        masked_b64 = base64.b64encode(image_file.read()).decode('utf-8')
                
                curr_masked_list.append(masked_b64)
            
            all_masked_images.append(curr_masked_list)
        
        return all_masked_images, activations[:, :self.n_exemplars], thresholds

class SyntheticExemplars:
    """
    A class for analyzing synthetic exemplars in neural networks.
    
    This class processes and stores synthetic exemplar images along with their
    activations and thresholds, organizing them in a nested dictionary structure
    similar to DatasetExemplars.

    Attributes
    ----------
    path2exemplars : str
        Path to the directory containing the exemplar images.
    path2save : str
        Path to the directory where the results will be saved.
    mode : str
        The mode of synthetic exemplar generation.
    n_exemplars : int
        Number of exemplar images to use for each unit.
    im_size : int
        Size to which the images will be resized.
    exemplars : defaultdict
        Nested dictionary storing exemplar images: mode->layer->unit.
    activations : defaultdict
        Nested dictionary storing unit activations: mode->layer->unit.
    thresholds : defaultdict
        Nested dictionary storing unit thresholds: mode->layer->unit.
    """
    
    def __init__(self, path2exemplars: str, path2save: str, mode: str, n_exemplars: int = 15, im_size: int = 224):
        """
        Initialize the SyntheticExemplars object.

        Parameters
        ----------
        path2exemplars : str
            Path to the directory containing the exemplar images.
        path2save : str
            Path to the directory where the results will be saved.
        mode : str
            The mode of synthetic exemplar generation.
        n_exemplars : int, optional
            Number of exemplar images to use for each unit (default is 15).
        im_size : int, optional
            Size to which the images will be resized (default is 224).
        """
        self.path2exemplars = path2exemplars
        self.n_exemplars = n_exemplars
        self.path2save = path2save
        self.im_size = im_size
        self.mode = mode

        # Initialize nested defaultdict structures
        self.exemplars = defaultdict(lambda: defaultdict(dict))
        self.activations = defaultdict(lambda: defaultdict(dict))
        self.thresholds = defaultdict(lambda: defaultdict(dict))

        # Process the exemplars
        exemplars, activations = self.net_dissect()
        
        # Store results in nested structure
        self.exemplars["synthetic"][mode] = exemplars
        self.activations["synthetic"][mode] = activations


    def net_dissect(self,im_size=224):
        exp_path = os.path.join(self.path2exemplars, self.mode)
        activations = np.loadtxt(os.path.join(exp_path, 'activations.csv'), delimiter=',')
        image_array = np.load(os.path.join(exp_path, 'images.npy'))
        mask_array = np.load(os.path.join(exp_path, 'masks.npy'))
        all_images = []
        for unit in range(activations.shape[0]):
            curr_image_list = []
            for exemplar_inx in range(min(activations.shape[1],self.n_exemplars)):
                save_path = os.path.join(self.path2save,'synthetic_exemplars',self.mode,str(unit))
                if os.path.exists(os.path.join(save_path,f'{exemplar_inx}.png')):
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(masked_image)
                else:
                    curr_mask = (mask_array[unit,exemplar_inx]/255)+0.25
                    curr_mask[curr_mask>1]=1
                    curr_image = image_array[unit,exemplar_inx]
                    masked_image = curr_image * curr_mask
                    masked_image =  Image.fromarray(masked_image.astype(np.uint8))
                    os.makedirs(save_path,exist_ok=True)
                    masked_image.save(os.path.join(save_path,f'{exemplar_inx}.png'), format='PNG')
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(masked_image)
            all_images.append(curr_image_list)

        return all_images, activations[:,:self.n_exemplars]
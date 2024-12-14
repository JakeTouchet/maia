import argparse
import os
import sys
import random
import torch
import openai
from dotenv import load_dotenv
from datetime import datetime
import traceback

# Load environment variables
load_dotenv()
# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

from maia_api import System, Tools, SyntheticSystem
from utils.DatasetExemplars import DatasetExemplars, SyntheticExemplars
from utils.main_utils import load_unit_config, generate_save_path
from utils.InterpAgent import InterpAgent

import json
from IPython import embed

random.seed(0000)

def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')    
    parser.add_argument('--maia', type=str, default='gpt-4o-mini', choices=['gpt-4-vision-preview','gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini'], help='maia agent name')    
    parser.add_argument('--task', type=str, default='neuron_description', help='task to solve, default is unit description')
    parser.add_argument('--synthetic', action='store_true', help='Runs maia with synthetic neurons', default=False)
    parser.add_argument('--unit_config_name', type=str, default='test.json', help='name of unit config file defining what units to test')    
    parser.add_argument('--n_exemplars', type=int, default=15, help='number of examplars for initialization')    
    parser.add_argument('--images_per_prompt', type=int, default=1, help='number of images per prompt')    
    parser.add_argument('--path2save', type=str, default='./results', help='a path to save the experiment outputs')    
    parser.add_argument('--path2prompts', type=str, default='./prompts', help='path to prompt to use')    
    parser.add_argument('--path2exemplars', type=str, default='./exemplars', help='path to net disect top 15 exemplars images')    
    parser.add_argument('--device', type=int, default=0, help='gpu device to use (e.g. 1)')    
    parser.add_argument('--text2image', type=str, default='sd', choices=['sd','dalle'], help='name of text2image model')    
    parser.add_argument('--debug', action='store_true', help='debug mode, print dialogues to screen', default=False)
    # Manual Unit Definition
    parser.add_argument('--model', type=str, default='resnet152', help='name of the first model')
    parser.add_argument('--layer', type=str, default='layer4', help='layer to interpret')
    parser.add_argument('--neurons', nargs='+', type=int, help='list of neurons to interpret')
    return parser.parse_args()

def setup_error_logging(path2save):
    """Set up error logging to file"""
    log_dir = os.path.join(path2save, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'error_log_{timestamp}.txt')
    return log_file

def log_error(error, log_file, context=""):
    """Log error with traceback to file and print to console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    error_message = f"\n{'='*50}\n"
    error_message += f"Timestamp: {timestamp}\n"
    if context:
        error_message += f"Context: {context}\n"
    error_message += f"Error: {str(error)}\n"
    error_message += f"Traceback:\n{traceback.format_exc()}\n"
    
    print(error_message)
    
    with open(log_file, 'a') as f:
        f.write(error_message)

def main(args):
    path2save = args.path2save
    log_file = None
    
    try:
        if args.unit_config_name is not None:
            unit_config = load_unit_config(args.unit_config_name)
        else:
            unit_config = {args.model: {args.layer: args.neurons}}

        path2save = generate_save_path(path2save, args.maia, args.unit_config_name or "manual_config")
        print(path2save)
        os.makedirs(path2save, exist_ok=True)
        
        # Set up error logging
        log_file = setup_error_logging(path2save)
        
        # Initialize MAIA agent
        maia = InterpAgent(
            model_name=args.maia,
            prompt_path=args.path2prompts,
            api_prompt_name="api.txt",
            user_prompt_name=f"user_{args.task}.txt",
            overload_prompt_name="final.txt",
            end_experiment_token="[FINAL]",
            max_round_count=25,
            debug=args.debug
        )

        # Loop through all defined units
        for model in unit_config.keys():
            for layer in unit_config[model].keys():
                for neuron_num in unit_config[model][layer]:
                    if model == "synthetic":
                        net_dissect = SyntheticExemplars(
                            os.path.join(args.path2exemplars, model),
                            args.path2save,
                            layer
                        )
                        with open(os.path.join('./synthetic_neurons_dataset', "labels", f'{layer}.json'), 'r') as file:
                            synthetic_neuron_data = json.load(file)
                            embed()
                            gt_label = synthetic_neuron_data[neuron_num]["label"].rsplit('_')
                        system = SyntheticSystem(neuron_num, gt_label, layer, args.device)
                    else:
                        net_dissect = DatasetExemplars(
                            args.path2exemplars,
                            args.n_exemplars,
                            args.path2save,
                            unit_config
                        )
                        system = System(model, layer, neuron_num, net_dissect.thresholds, args.device)

                    tools = Tools(
                        path2save,
                        args.device,
                        maia,
                        system,
                        net_dissect,
                        images_per_prompt=args.images_per_prompt,
                        text2image_model_name=args.text2image,
                        image2text_model_name=args.maia
                    )

                    # Run all tests with error handling
                    test_functions = [
                        ("test_call_neuron_single", test_call_neuron_single),
                        ("test_call_neuron_with_edit", test_call_neuron_with_edit),
                        ("test_dataset_exemplars", test_dataset_exemplars),
                        ("test_edit_images_dog_cat", test_edit_images_dog_cat),
                        ("test_edit_images_dog_actions", test_edit_images_dog_actions),
                        ("test_text2image", test_text2image),
                        ("test_display_single", test_display_single),
                        ("test_display_multiple", test_display_multiple),
                        ("test_summarize_images", test_summarize_images),
                        ("test_describe_images", test_describe_images)
                    ]
                    
                    for test_name, test_func in test_functions:
                        try:
                            print(f"\nRunning {test_name}...")
                            test_func(tools, system)
                            print(f"{test_name} completed successfully")
                        except Exception as e:
                            log_error(e, log_file, f"Error in test function: {test_name}")
                            continue

                    # Generate HTML report
                    try:
                        tools.generate_html(maia.experiment_log)
                        print("HTML generation completed successfully")
                    except Exception as e:
                        log_error(e, log_file, "Error in HTML generation")
    
    except Exception as e:
        if log_file:
            log_error(e, log_file, "Error in main initialization")
        else:
            print(f"Error in main initialization (before log file creation):")
            print(traceback.format_exc())

# [Previous test functions remain unchanged]
def test_call_neuron_single(tools: Tools, system: System):
    """Test call_neuron with a single prompt"""
    prompt = ["a dog standing on the grass"]
    image = tools.text2image(prompt)
    activations, images = system.call_neuron(image)
    for activation, image in zip(activations, images):
        tools.display(image, f"Activation: {activation}")

def test_call_neuron_with_edit(tools: Tools, system: System):
    """Test call_neuron with image editing"""
    prompt = ["a dog standing on the grass"]
    edits = ["replace the dog with a lion"]
    all_images, all_prompts = tools.edit_images(prompt, edits)
    activation_list, image_list = system.call_neuron(all_images)
    for activation, image in zip(activation_list, image_list):
        tools.display(image, f"Activation: {activation}")

def test_dataset_exemplars(tools, system):
    """Test dataset_exemplars functionality"""
    activations, images = tools.dataset_exemplars()
    for activation, image in zip(activations, images):
        tools.display(image, f"Activation: {activation}")

def test_edit_images_dog_cat(tools: Tools, system: System):
    """Test edit_images with dog to cat transformation"""
    prompt = ["a dog standing on the grass"]
    edits = ["replace the dog with a cat"]
    all_images, all_prompts = tools.edit_images(prompt, edits)
    activation_list, image_list = system.call_neuron(all_images)
    for activation, image in zip(activation_list, image_list):
        tools.display(image, f"Activation: {activation}")

def test_edit_images_dog_actions(tools, system):
    """Test edit_images with different dog actions"""
    prompts = ["a dog standing on the grass"]*3
    edits = ["make the dog sit","make the dog run","make the dog eat"]
    all_images, all_prompts = tools.edit_images(prompts, edits)
    activation_list, image_list = system.call_neuron(all_images)
    for activation, image in zip(activation_list, image_list):
        tools.display(image, f"Activation: {activation}")

def test_text2image(tools, system):
    """Test text2image functionality"""
    prompt_list = ["a dog standing on the grass", 
                   "a dog sitting on a couch",
                   "a dog running through a field"]
    images = tools.text2image(prompt_list)
    activation_list, image_list = system.call_neuron(images)
    for activation, image in zip(activation_list, image_list):
        tools.display(image, f"Activation: {activation}")

def test_display_single(tools, system):
    """Test display with a single image"""
    prompt = ["a dog standing on the grass"]
    images = tools.text2image(prompt)
    activation_list, image_list = system.call_neuron(images)
    for activation, image in zip(activation_list, image_list):
        tools.display(image, f"Activation: {activation}")

def test_display_multiple(tools, system):
    """Test display with multiple images"""
    prompt_list = ["a dog standing on the grass", 
                   "a dog sitting on a couch",
                   "a dog running through a field"]
    images = tools.text2image(prompt_list)
    activation_list, image_list = system.call_neuron(images)
    for activation, image in zip(activation_list, image_list):
        tools.display(image, f"Activation: {activation}")

def test_summarize_images(tools: Tools, system: System):
    """Test summarize_images functionality"""
    _, exemplars = tools.dataset_exemplars()
    summarization = tools.summarize_images(exemplars)
    tools.display(summarization)

def test_describe_images(tools: Tools, system: System):
    """Test describe_images functionality"""
    prompt_list = ["a dog standing on the grass", 
                   "a dog sitting on a couch",
                   "a dog running through a field"]
    images = tools.text2image(prompt_list)
    _, image_list = system.call_neuron(images)
    descriptions = tools.describe_images(image_list, prompt_list)
    tools.display(descriptions)

if __name__ == "__main__":
    args = call_argparse()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    main(args)
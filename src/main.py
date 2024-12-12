import argparse
import os
import sys
import random
import torch

# Simplest way to integrate the VisDiff repository
sys.path.append(os.path.abspath('VisDiff'))

from maia_api import System, Tools
from utils.DatasetExemplars import DatasetExemplars
from utils.main_utils import load_unit_config, generate_save_path, create_unit_config
from utils.CodeAgent import InterpAgent

random.seed(0000)

layers = {
    "resnet152": ['conv1','layer1','layer2','layer3','layer4'],
    "clip-RN50" : ['layer1','layer2','layer3','layer4'],
    "dino_vits8": ['blocks.1.mlp.fc1','blocks.3.mlp.fc1','blocks.5.mlp.fc1','blocks.7.mlp.fc1','blocks.9.mlp.fc1','blocks.11.mlp.fc1'],
    "synthetic_neurons": ['mono','or','and']
}

def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')	
    parser.add_argument('--maia', type=str, default='gpt-4o', choices=['gpt-4-vision-preview','gpt-4-turbo', 'gpt-4o'], help='maia agent name')	
    parser.add_argument('--task', type=str, default='survey', help='task to solve, default is unit description')
    parser.add_argument('--model', type=str, default='resnet152', help='name of the first model')
    parser.add_argument('--layer', type=str, default='layer4', help='layer to interpret')
    parser.add_argument('--neurons', nargs='+', type=int, help='list of neurons to interpret')
    parser.add_argument('--n_exemplars', type=int, default=15, help='number of examplars for initialization')	
    parser.add_argument('--images_per_prompt', type=int, default=1, help='number of images per prompt')	
    parser.add_argument('--path2save', type=str, default='./results/', help='a path to save the experiment outputs')	
    parser.add_argument('--path2prompts', type=str, default='./prompts/', help='path to prompt to use')	
    parser.add_argument('--path2exemplars', type=str, default='./exemplars/', help='path to net disect top 15 exemplars images')	
    parser.add_argument('--device', type=int, default=0, help='gpu device to use (e.g. 1)')	
    parser.add_argument('--text2image', type=str, default='sd', choices=['sd','dalle'], help='name of text2image model')	
    parser.add_argument('--debug', action='store_true', help='debug mode, print dialogues to screen', default=False)
    return parser.parse_args()

# TODO - Re-add manual configs
def main(args):
    if args.unit_config_name is not None:
        unit_config_name = args.unit_config_name
        num_experiments = 1
    else:
        num_experiments = len(args.neurons)
    
    for i in range(num_experiments):
        unit_config = load_unit_config(unit_config_name)
        
        path2save = generate_save_path(args.path2save, args.maia, unit_config_name)
        os.makedirs(path2save, exist_ok=True)

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
        net_dissect = DatasetExemplars(args.path2exemplars, args.n_exemplars, args.path2save, unit_config)

        # Loop through all defined units
        for model, layer in unit_config.items():
            for neuron_num in unit_config[model][layer]:
                system = System(model, layer, neuron_num, net_dissect.thresholds, args.device)
                tools = Tools(path2save, 
                            args.device, 
                            maia, 
                            system, 
                            net_dissect, 
                            images_per_prompt=args.images_per_prompt, 
                            text2image_model_name=args.text2image, 
                            image2text_model_name=args.maia)

                maia.run_experiment(system, tools, save_html=True)
        
        # Save the dialogue
        with open(os.path.join(path2save, "experiment_log.json"), "w") as f:
            f.write(str(maia.experiment_log))

if __name__ == '__main__':
    args = call_argparse()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    main(args)
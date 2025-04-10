import argparse
import os
import random
import torch
import gc
import time
from maia_api import System, Tools, SyntheticSystem
from utils.DatasetExemplars import DatasetExemplars, SyntheticExemplars
from utils.main_utils import load_unit_config, generate_save_path, retrieve_synth_label, str2dict, create_bias_prompt
from utils.InterpAgent import InterpAgent

random.seed(0000)

def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')	
    parser.add_argument('--maia', type=str, default='gpt-4o', choices=['gpt-4-vision-preview','gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'claude'], help='maia agent name')	
    parser.add_argument('--task', type=str, default='neuron_description', help='task to solve, default is neuron description', choices=['neuron_description', 'bias_discovery'])
    parser.add_argument('--unit_config_name', type=str, help='name of unit config file defining what units to test. Leave blank for manual definition')	
    parser.add_argument('--n_exemplars', type=int, default=15, help='number of examplars for initialization')	
    parser.add_argument('--images_per_prompt', type=int, default=1, help='number of images per prompt')	
    parser.add_argument('--path2save', type=str, default='./results', help='a path to save the experiment outputs')	
    parser.add_argument('--path2prompts', type=str, default='./prompts', help='path to prompt to use')	
    parser.add_argument('--path2exemplars', type=str, default='./exemplars', help='path to net disect top 15 exemplars images')	
    parser.add_argument('--path2indices', type=str, default='./neuron_indices', help='path to neuron indices')
    parser.add_argument('--device', type=int, default=0, help='gpu device to use (e.g. 1)')	
    parser.add_argument('--text2image', type=str, default='sd', choices=['sd','dalle'], help='name of text2image model')	
    parser.add_argument('--debug', action='store_true', help='debug mode, print dialogues to screen', default=False)
    # Manual Unit Definition
    parser.add_argument('--model', type=str, default='resnet152', help='name of the model')
    parser.add_argument('--units', type=str2dict, default='layer4=122', help='units to interp')
    return parser.parse_args()

def main(args):
    path2save = args.path2save
    
    if args.unit_config_name is not None:
        unit_config = load_unit_config(args.path2indices, args.unit_config_name)
    else:
        unit_config = {args.model: args.units}

    # Loop through all defined units
    for model in unit_config.keys():
        for layer in unit_config[model].keys():
            for neuron_num in unit_config[model][layer]:
                try:
                    print(f"Running experiment on: {model}, {layer}, {neuron_num}")
                    # Save path should be path2save/maia-model/model/layer/neuron-num
                    path2save = os.path.join(args.path2save, args.maia, model, layer, str(neuron_num))
                    os.makedirs(path2save, exist_ok=True)
                    print("Save path:", path2save)

                    # Prompt needs to be created dynamically for bias_discovery so class label can be inserted
                    if args.task == "bias_discovery":
                        create_bias_prompt(args.path2indices, args.path2prompts, str(neuron_num))

                    # Initialize MAIA agent
                    print("Initializing MAIA...")
                    api = [
                        (System, [System.call_neuron]),
                        (Tools, [Tools.text2image, Tools.edit_images, Tools.dataset_exemplars, Tools.display, Tools.describe_images, Tools.summarize_images])
                    ]
                    maia = InterpAgent(
                        model_name=args.maia,
                        api=api,
                        prompt_path=args.path2prompts,
                        api_prompt_name="api.txt",
                        user_prompt_name=f"user_{args.task}.txt",
                        overload_prompt_name="final.txt",
                        end_experiment_token="[DESCRIPTION]", # IMPORTANT: Make sure this matches end-token specified by user prompt
                        max_round_count=15,
                        debug=args.debug
                    )
                    if model == "synthetic":
                        net_dissect = SyntheticExemplars(
                            os.path.join(args.path2exemplars, model),
                            args.path2save,
                            layer
                        )
                        gt_label = retrieve_synth_label(layer, neuron_num)
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

                    maia.run_experiment(system, tools, save_html=True)
                finally:
                    # Save the dialogue and the final description
                    print("Saving experiment log...")
                    with open(os.path.join(path2save, "experiment_log.json"), "w") as f:
                        f.write(str(maia.experiment_log))
                    try:
                        with open(os.path.join(path2save, "description.txt"), "w") as f:
                            f.write(str(maia.experiment_log[-1]['content'][0]['text']))
                    except:
                        print("No description found")

                    # Cleanup
                    print("Cleaning up...")
                    if 'tools' in locals():
                        tools.cleanup()
                        del tools
                    if 'system' in locals():
                        system.cleanup()
                        del system
                    
                    # Force garbage collection
                    gc.collect()
                    time.sleep(1)


if __name__ == '__main__':
    args = call_argparse()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    main(args)

'''Utils for the main.py file'''
import os
import json

# Convert a comma-separated key=value pairs into a dictionary
def str2dict(arg_value):
    my_dict = {}
    if arg_value:
        for item in arg_value.split(':'):
            key, value = item.split('=')
            values = value.split(',')
            my_dict[key] = [int(v) for v in values]
    return my_dict

# return the prompt according to the task
def return_prompt(prompt_path,setting='unit_description'):
    with open(f'{prompt_path}/api.txt', 'r') as file:
        sysPrompt = file.read()
    with open(f'{prompt_path}/user_{setting}.txt', 'r') as file:
        user_prompt = file.read()
    return sysPrompt, user_prompt

def create_bias_prompt(
        path2indices:str, 
        path2prompts:str, 
        unit:str, 
        base_prompt_name:str="user_bias_discovery_base.txt", 
        final_prompt_name:str="user_bias_discovery.txt"
        ):
    """
    Creates and saves a bias prompt for a given unit. 
    Loads the base prompt from user_bias_discovery_base.txt
    and saves the final prompt to user_bias_discovery.txt.
    """
    # Load class index and base prompt
    with open(os.path.join(path2indices, "imagenet_class_index.json"), 'r') as file:
        imagenet_classes = json.load(file)
    with open(os.path.join(path2prompts, base_prompt_name), 'r') as file:
        base_prompt = file.read()
    # Create the final prompt
    prompt = base_prompt.format(imagenet_classes[unit][1].replace("_"," "))
    # Save the final prompt
    with open(os.path.join(path2prompts, final_prompt_name), 'w') as file:
        file.write(prompt)
    file.close()

# save the field from the history to a file
def save_field(history, filepath, field_name, first=False, end=True):
    text2save = None
    for i in history:
        if i['role'] == 'assistant':
            for entry in i['content']:
                if field_name in entry["text"]:
                    if end:
                        text2save = entry["text"].split(field_name)[-1].split('\n')[0]
                    else:
                        text2save = entry["text"].split(field_name)[-1]
                    if first: break
    if text2save!= None:
        with open(filepath,'w') as f:
            f.write(text2save) 
        f.close()   

# save the full experiment history to a .json file
def save_history(history,filepath):
    with open(filepath+'.json', 'w') as file:
        json.dump(history, file)
    file.close()

# save the full experiment history, and the final description and label to .txt files.
def save_dialogue(history,path2save):
    save_history(history,path2save+'/history')
    save_field(history, path2save+'/description.txt', '[DESCRIPTION]: ')
    save_field(history, path2save+'/label.txt', '[LABEL]: ', end=True)

# final instructions to maia
def overload_instructions(tools, prompt_path='./prompts/'):
    with open(f'{prompt_path}/final.txt', 'r') as file:
        final_instructions = file.read()
        tools.update_experiment_log(role='user', type="text", type_content=final_instructions)

def load_unit_config(path2indices:str, unit_file_name:str):
    with open(os.path.join(path2indices, unit_file_name)) as json_file:
        unit_config = json.load(json_file)
    return unit_config

def generate_save_path(path2save: str, maia: str, unit_file_name: str):
    path2save = os.path.join(path2save, maia, unit_file_name)
    return path2save

def generate_numbered_path(path:str, file_extension:str=""):
    i = 0
    numbered_path = path + "_" + str(i) + file_extension
    while os.path.exists(numbered_path):
        i += 1
        numbered_path = path + "_" + str(i) + file_extension
    return numbered_path

def retrieve_synth_label(layer, neuron_num):
    with open(os.path.join('./synthetic_neurons_dataset', "labels", f'{layer}.json'), 'r') as file:
        synthetic_neuron_data = json.load(file)
        gt_label = synthetic_neuron_data[neuron_num]["label"].rsplit('_')
    return gt_label

# A Multimodal Automated Interpretability Agent #
### ICML 2024 ###

### [Project Page](https://multimodal-interpretability.csail.mit.edu/maia) | [Arxiv](https://multimodal-interpretability.csail.mit.edu/maia) | [Experiment browser](https://multimodal-interpretability.csail.mit.edu/maia/experiment-browser/)

<img align="right" width="42%" src="/docs/static/figures/maia_teaser.jpg">

[Tamar Rott Shaham](https://tamarott.github.io/)\*, [Sarah Schwettmann](https://cogconfluence.com/)\*, <br>
[Franklin Wang](https://frankxwang.github.io/), [Achyuta Rajaram](https://twitter.com/AchyutaBot), [Evan Hernandez](https://evandez.com/), [Jacob Andreas](https://www.mit.edu/~jda/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/) <br>
\*equal contribution <br><br>

MAIA is a system that uses neural models to automate neural model understanding tasks like feature interpretation and failure mode discovery. It equips a pre-trained vision-language model with a set of tools that support iterative experimentation on subcomponents of other models to explain their behavior. These include tools commonly used by human interpretability researchers: for synthesizing and editing inputs, computing maximally activating exemplars from real-world datasets, and summarizing and describing experimental results. Interpretability experiments proposed by MAIA compose these tools to describe and explain system behavior.

**News** 
[December ?]: Released MAIA.2
\
[July 3]: We release MAIA implementation code for neuron labeling 
\
[August 14]: Synthetic neurons are now available (both in `demo.ipynb` and in `main.py`)

**This repo is under active development. Sign up for updates by email using [this google form](https://forms.gle/Zs92DHbs3Y3QGjXG6).**

## Updates
- [x] Rework MAIA’s code execution to run multiple free-form blocks of code, rather than having MAIA define a single function that is then executed.
- [x] Rework MAIA to use a flexible display function which recursively unpacks passed iterables and append them to MAIA’s experiment log.
- [x] Refactor MAIA’s api file to be purely front-facing, moving back-end functions to various utils files. Public functions that MAIA executes now follow standard OOP naming conventions. Simplify main.py, utils, and neuron configuration.
- [x] Re-implement MAIA as an InterpAgent. The InterpAgent class allows the creation of tools as sub-agents which can execute code given a user and api prompt.
- [x] Add gpt-4o, gpt-4o-mini, and gpt-4-turbo.
- [x] Add ability to automatically generate api.txt file from passed functions and classes, removing the need to copy and paste documentation changes


### Installations ###
clone this repo and install all requirements:
```bash
git clone https://github.com/multimodal-interpretability/maia.git
cd maia
bash setup_env.sh
```

download [net-dissect](https://netdissect.csail.mit.edu/) precomputed exemplars:
```bash
bash download_exemplars.sh
```

## Quick Start ##
You can run demo experiments on individual units using ```demo.ipynb```:
\
\
Install Jupyter Notebook via pip (if Jupyter is already installed, continue to the next step)
```bash
pip install notebook
```
Launch Jupyter Notebook
```bash
jupyter notebook
```
This command will start the Jupyter Notebook server and open the Jupyter Notebook interface in your default web browser. The interface will show all the notebooks, files, and subdirectories in this repo (assuming is was initiated from the maia path). Open ```demo.ipynb``` and proceed according to the instructions.

NEW: `demo.ipynb` now supports synthetic neurons. Follow installation instructions at `./synthetic-neurons-dataset/README.md`. After installation is done, you can define MAIA to run on synthetic neurons according to the instructions in `demo.ipynb`.

### Batch experimentation ###
To run a batch of experiments, use ```main.py```:

#### Load openai/huggingface api key ####
(in case you don't have these api-keys, you can follow the instructions [here](https://platform.openai.com/docs/quickstart) and [here]https://huggingface.co/docs/hub/security-tokens).

For huggingface, you will need access to stable-diffusion-3.5, which can be obtained [here]https://huggingface.co/stabilityai/stable-diffusion-3.5-medium.

Set your api-keys as an environment variables (this is a bash command, look [here](https://platform.openai.com/docs/quickstart) for other OS)
```bash
export OPENAI_API_KEY='your-api-key-here'
export HF_TOKEN='your-api-key-here'
```

#### Run MAIA ####
Manually specify the model and desired units in the format ```layer#1=unit#1,unit#2... : layer#1=unit#1,unit#2...``` by calling e.g.:
```bash
python main.py --model resnet152 --units layer3=229,288:layer4=122,210
``` 
OR by loading a ```.json``` file specifying the units (see example in ```./neuron_indices/```)
```bash
python main.py --unit_config_name resnset152.json
```
Adding ```--debug``` to the call will print all results to the screen.

To run maia using another Multi-Modal LLM like Claude, use ```--maia model_name```. Ex:
```bash
python main.py --maia claude-3-7-sonnet-latest  --unit_config_name resnset152.json
``` 

Refer to the documentation of ```main.py``` for more configuration options.

Results are automatically saved to an html file under ```./results/``` and can be viewed in your browser by starting a local server:
```bash
python -m http.server 80
```
Once the server is up, open the html in [http://localhost:80](http://localhost:80/results/)

## Additional Configuration ##

#### Run MAIA on sythetic neurons ####

You can now run maia on synthetic neurons with ground-truth labels (see sec. 4.2 in the paper for more details).

Follow installation instructions at `./synthetic-neurons-dataset/README.md`. Then you should be able to run `main.py` on synthetic neurons by calling e.g.:
```bash
python main.py --model synthetic_neurons --unit_mode manual --units mono=1,8:or=9:and=0,2,5
``` 
(neuron indices are specified according to the neuron type: "mono", "or" and "and").

You can also use the .json file to run all synthetic neurons (or specify your own file):
```bash
python main.py --model synthetic_neurons --unit_mode from_file --unit_file_path ./neuron_indices/
```

### Running netdissect

MAIA uses pre-computed exemplars for its experiments. Thus, to run MAIA on a new model, you must first compute the exemplars netdissect. MAIA was built to use exemplar data as generated by MILAN’s implementation of netdissect. 

First, clone MILAN.

```python
git clone https://github.com/evandez/neuron-descriptions.git
```

Then, set the environment variables to control where the data is inputted/outputted. Ex:

```python
%env MILAN_DATA_DIR=./data
%env MILAN_MODELS_DIR=./models
%env MILAN_RESULTS_DIR=./results
```

Make those three directories mentioned above.

When you have a new model you want to dissect:

1. Make a folder in `models` directory
2. Name it whatever you want to call your model (in this case we'll use `resnet18syn`)
3. Inside that folder, place your saved model file, but name it `imagenet.pth` (since you'll be dissecting with ImageNet)

Now, place ImageNet in the data directory. If you have imagenet installed, you create a symbolic link to it using:

```bash
ln -s /path/to/imagenet /path/to/data/imagenet
```

If you don’t have imagenet installed, follow the download instructions [here](https://image-net.org/download.php): 

Next, add under your keys in `models.py` (located in `src/exemplars/models.py`):

```python
KEYS.RESNET18_SYNTHETIC = 'resnet18syn/imagenet'
```

Then add this in the keys dict:

```python
KEYS.RESNET18_SYNTHETIC:
    ModelConfig(
        models.resnet18,
        load_weights=True,
        layers=LAYERS.RESNET18,
    ),
```

Once that's done, you should be ready to run the compute exemplars script:

```bash
python3 -m scripts.compute_exemplars resnet18syn imagenet --device cuda
```

This will run the compute exemplars script using the ResNet18 synthetic model on the ImageNet dataset, utilizing CUDA for GPU acceleration.

Finally, move the computed exemplars to the `exemplars/` folder.

### To set up the synthetic neurons:


1. **Follow the setup instructions on Grounded-SAM setup:**
   - Export global variables (choose whether to run on CPU or GPU; note that running on CPU is feasible but slower, approximately 3 seconds per image):
     ```bash
     export AM_I_DOCKER=False
     export BUILD_WITH_CUDA=True
     export CUDA_HOME=/path/to/cuda-11.3/

     ```
   - Install osx:
     ```bash
     cd grounded-sam-osx && bash install.sh
     ```
     
2. **Download grounded DINO and grounded SAM .pth files**  
   - Download groudned DINO: 
     ```bash
     cd .. #back to ./Grounded_Segment-Anything
     #download the pretrained groundingdino-swin-tiny model
     wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
     ```
   - Download grounded SAM: 
     ```bash
     wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
     ```
    - Try running grounded SAM demo:
      ```bash
      export CUDA_VISIBLE_DEVICES=0
      python grounded_sam_demo.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --input_image assets/demo1.jpg \
        --output_dir "outputs" \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "bear" \
        --device "cpu"
      ```
  
### Note
Initially Grounded Segment Anything was implemented as a git submodule. However, a commit broke synthetic neuron functionality. Therefore, it is now
simply a hard-copied version of commit 2b1b72e to avoid further commits affecting the repository.

## Creating Custom Synthetic-Neurons

1. Decide the mode and label(s) for your neuron.
    1. ex. Cheese OR Lemon
2. Identify an object-classification dataset you want to create exemplars from. The example notebooks use COCO.
3. Query the dataset for relevant images to your synthetic neuron
    1. i.e. if you want to build a neuron that’s selective for dogs (monosemantic neuron), query the dataset for dog images
    2. If you want to build a neuron that’s selective for dogs but only if they’re wearing collars (and neuron), query the dataset for dogs wearing collars
4. Instantiate synthetic neuron with desired setting and labels
5. Run SAMNeuron on all candidate images and save the top 15 highest activating images and their activations
6. Save the data in the format “path2data/label” or for multi-label neurons “path2data/label1_label2”
7. Convert the saved data to the format expected by MAIA, demonstrated [here](synthetic_neurons_dataset/create_synthetic_neurons.ipynb)

### Acknowledgment ###
[Christy Li](https://www.linkedin.com/in/christykl/) helped with cleaning up the synthetic neurons code for release.

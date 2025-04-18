# Setup Instructions for Synthetic Neurons

## To set up the synthetic neurons:


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

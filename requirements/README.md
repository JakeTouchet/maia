# MAIA Environment Setup

This guide describes how to set up the environment for the MAIA project, including dependencies and model checkpoints for SAM (Segment Anything Model) and Grounding DINO.

---

## 🐍 Create Conda Environment

```bash
conda create -n maia_env python=3.10 --file conda_packages.txt -c nvidia
conda activate maia_env
```
---

## 📦 Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -r torch_requirements.txt --force-reinstall
```

---

## ⚙️ Environment Variables for SAM and Grounding DINO

```bash
export AM_I_DOCKER="False"
export BUILD_WITH_CUDA="True"
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CC=$(which gcc-12)
export CXX=$(which g++-12)
```

---

## 📥 Install SAM and Grounding DINO

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

---

## 📦 Download Model Checkpoints

```bash
# Segment Anything (SAM)
wget "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Grounding DINO
wget "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
```

---

## ✅ Notes

- Ensure you have `gcc-12` and `g++-12` installed on your system.
- If you're running in Docker, set `AM_I_DOCKER="True"` instead.
- These steps assume CUDA is correctly installed and `nvcc` is available in your path.
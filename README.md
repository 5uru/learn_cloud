# learn_cloud

# CNN Training on Cloud GPU

This repository contains a CNN model implementation using tinygrad for MNIST classification, with training performance tracking via trackio (Hugging Face's alternative to Weights & Biases).

## Why This Approach?

When learning deep learning, local machines often struggle with performance or take too long for training. While Google Colab is an option, I prefer coding in my own environment (PyCharm). Cloud-based GPU solutions provide the best of both worlds - coding locally while executing on powerful remote machines.

## Setup Instructions

### 1. Generate SSH Key
```bash
ssh-keygen -t ed25519  # Press Enter 3 times
```

### 2. Get Your Public Key
```bash
cat ~/.ssh/id_ed25519.pub
```

### 3. Add SSH Key to GPU Droplet
- Go to your cloud provider (e.g., DigitalOcean)
- Find your droplet's IP in the networking tab
- Add your SSH key to the droplet

### 4. Connect to Your Droplet
```bash
ssh root@YOUR_DROPLET_IP
```
- Type "yes" when prompted
- You can now use FileZilla for file transfers if needed

### 5. Set Up Environment
```bash
apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 6. Run the CNN Training
```bash
python cnn.py
```

## Project Details

The CNN implementation (`cnn.py`) includes:
- A simple convolutional neural network for MNIST classification
- Integration with trackio for experiment tracking
- Automatic model saving and uploading to Hugging Face Hub
- Performance monitoring and best model tracking

This setup enables efficient deep learning experimentation while maintaining a comfortable local development workflow.
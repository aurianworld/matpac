# MATPAC Inference Code

In this folder we propose an easy to install python package to use MATPAC
at inference (or for fine-tuning) as a feature extractor from raw audio.

## Installation

Place yourself in the folder [inference_matpac](.), then install the package
in editable mode :

```bash
pip install -e .
```

## Usage 

Download the model [weights](https://github.com/aurianworld/matpac/releases/download/Initial_release/matpac_10_2048.pt) and save them under `ckpt_path` on your computer.

### Basic Usage

```python
import torchaudio
from matpac.model import get_matpac

# Load audio (ensure mono; stereo channels are treated as separate batch dimensions. Ensure that its sample rate is 16000Hz)

x, sr = torchaudio.load("my_file.wav")
x = x.mean(dim=0, keep_dim=True)  # MATPAC takes mono audio as input (bs, n_samples)

# Instantiate the model 
model = get_matpac(checkpoint_path=ckpt_path)

# Extract the features from the audio
emb, layer_results = model(x) 

# emb is the output of the last encoder layer. Shape (bs, 3840)
# layer_results is the output of each encoder layer. Shape (bs, 12, 3840)

```

### Advanced Usage

- By default we mean pool the time dimension of the embedding, but you can keep it
by passing this argument :

```python

# Instantiate the model without pooling the  temporal dimension
model = get_matpac(checkpoint_path=ckpt_path, pull_time_dimension=False)

# Extract the features from the audio
emb, layer_results = model(x) 

# emb : shape (bs, T, 3840)
# layer_results : shape (bs, 12, T, 3840)
```

- The default feature extraction is the one we used for the results of our paper, but it can be quite slow if you have a big batch size or if you want to fine tune the encoder. Therefore, we added faster but less precise, as it adds some padding values, feature extraction method.

```python

# Instantiate the model without pooling the  temporal dimension
model = get_matpac(checkpoint_path=ckpt_path, inference_type="fast")

# Extract the features from the audio
emb, layer_results = model(x) 

# emb : shape (bs, 3840)
# layer_results : shape (bs, 12, 3840)
```

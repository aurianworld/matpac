# MATPAC Inference Code

In this folder we propose an easy to install python package to use MATPAC
at inference (or for fine-tuning) as a feature extractor from raw audio.

## AVAILABLE WEIGHTS
All model weights are available in the release of the repository.

- MATPAC: matpac_10_2048.pt
- MATPAC++, SSL on AS: matpac_plus_6s_2048_enconly.pt
- MATPAC++, SSL on Music: matpac_plus_music_6s_2048_enconly.pt
- MATPAC++, SSL+FT on AS, encoder only: matpac_plus_6s_2048_enconly.pt
- MATPAC++, SSL+FT on AS, encoder+class head: matpac_plus_as_48_1_map_enc_and_head.pt

## Installation

Place yourself in the folder [inference_matpac](.), then install the package
in editable mode :

```bash
pip install -e .
```

## Embedding extraction 

Download the model [weights](https://github.com/aurianworld/matpac/releases/download/Initial_release/matpac_10_2048.pt) and save them under `ckpt_path` on your computer.

### Basic embedding extraction

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

### Advanced embedding extraction

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

## Classification heads and labels prediction

### AudioSet classification head

```python

# Instantiate the model with the AudioSet finetuning checkpoint
# Set as_class_head to True to use the AudioSet class predictions.
model = get_matpac(
            checkpoint_path="matpac_plus_as_48_1_map_enc_and_head.pt",
            as_class_head=True
            )

# Extract the probabilities and the associated labels from the audio
prob, labels = model.probe_forward(audio_real)

# prob : shape (bs, 527) is the logit vector on over the AudioSet Classes
# labels : shape (bs, 5) give the 5 most activated AudioSet classes and their probabilities.

labels -> [['Heart sounds, heartbeat (0.62)', 'Speech (0.18)', 'Heart murmur (0.14)', 'Throbbing (0.01)', 'Inside, small room (0.01)']]
```

### Downstream tasks classification heads

We release the weights of all the classification heads of each downstream tasks for each pre-trained MATPAC++ model. 

You can then make class predictions over any audio on the 11 downstream tasks used in the study. You can find the head's weights [here](https://github.com/aurianworld/matpac/releases/download/Probes_weights/Weights_probes_matpac.zip).

```python

# Instantiate the model with the pre-trained checkpoint and FSD50K classification head
model = get_matpac(
            checkpoint_path="matpac_plus_6s_2048_enconly.pt",
            probe_checkpoint_paths="matpac++_general_audio/fsd50k.pth"
            )

# Extract the probabilities and the associated labels from the audio
prob, labels = model.probe_forward(audio_real)

# prob : shape (bs, n_classes) is the logit vector on over the downstream task's classes
# labels : shape (bs, 5) give the 5 most activated AudioSet classes and their probabilities.

labels -> [['Domestic_sounds_and_home_sounds (0.39)', 'Tools (0.09)', 'Sawing (0.06)', 'Glass (0.04)', 'Animal (0.04)']]
```
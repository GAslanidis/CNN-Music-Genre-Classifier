# CNN Music Genre Classifier

A convolutional neural network that classifies music clips into **10 genres** by treating their mel spectrograms as images. Built with **TensorFlow/Keras** and **librosa**, trained on the GTZAN dataset.

Genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock.

## How it works

1. **Audio → image.** Each 30-second clip is loaded at 22,050 Hz and converted into a 128-band mel spectrogram (in decibels), resized to a 128×128 single-channel "image".
2. **Data augmentation.** Each clip produces five extra training samples:
   - time stretch (0.8× and 1.2× speed)
   - pitch shift (±2 semitones)
   - added Gaussian noise
3. **CNN classification.** The spectrograms are fed to a CNN that outputs a probability for each genre.

## Model

The project went from a simple baseline to an improved architecture through a series of experiments.

| | Baseline | Improved |
|---|---|---|
| Conv blocks | 3 (32 → 64 → 128) | 4 (32 → 64 → 96 → 128) |
| Regularization | Dropout 0.5 | Batch normalization, L2, dropout 0.3 |
| Head | Flatten + Dense 256 | Global average pooling + Dense 128 |
| Optimizer | Adam (default) | Adam, lr 1e-4 with step-decay schedule |
| Training | 30 epochs | Early stopping (patience 5) |
| Data | Original clips | Original + 5 augmented versions |

Experiments covered layer and neuron counts, optimizers (Adam, SGD, RMSprop), batch normalization, dropout, learning-rate scheduling, split ratios, and augmentation.

## Results

| Model | Test accuracy |
|---|---|
| Baseline CNN | 29.0% |
| Improved CNN | 95.66% |

> **Note on evaluation:** In the current version, augmentation runs *before* the train/test split, so augmented copies of the same song can land in both sets. The test set is also used for early stopping. Both inflate the reported accuracy. A corrected evaluation (splitting by song first, augmenting only the training set, and using a separate validation set) is in progress and will replace these numbers.

## Getting started

The notebook is designed for **Google Colab** (a GPU runtime is recommended).

1. Open `Music_Genre_Classification.ipynb` in Colab.
2. Run the cells in order. The dataset downloads automatically from Kaggle via `kagglehub`.

Install locally:

```bash
pip install tensorflow librosa numpy matplotlib seaborn scikit-learn tqdm kagglehub
```

> The first cell uses the `radnn` library to mount a course workspace in Colab. Skip it if you're running the notebook elsewhere.

### Using the trained model

```python
import tensorflow as tf
model = tf.keras.models.load_model("best_model.h5")
```

Inputs must be preprocessed the same way as in the notebook: a 128×128 mel spectrogram in dB, shaped `(1, 128, 128, 1)`.

## Repository contents

| File | Description |
|---|---|
| `Music_Genre_Classification.ipynb` | Full pipeline: data loading, preprocessing, augmentation, training, and evaluation |
| `best_model.h5` | Trained model weights |

## Dataset

[GTZAN Dataset – Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) on Kaggle: 1,000 30-second clips, 100 per genre. The dataset isn't included in this repo and is subject to its own terms.

## Tech stack

Python · TensorFlow / Keras · librosa · NumPy · scikit-learn · Matplotlib · Seaborn · Google Colab

## License

Released under the [MIT License](LICENSE).

## Author

**Georgios Aslanidis**. Developed as a semester project for a Computer Science degree at the American College of Thessaloniki.

#GAN for audio prediction and inpainting

## Creating the dataset
To create the dataset, use the script [dataset2tfrecord](python/dataset2tfrecord.py)
This script can be used to parse the meta-data of the MAESTRO or FMA dataset and create 3 tfrecord files. 
One is the training set, the other is the validation set and the last one is the testing set. 
These tfrecord has the feature audio which contains an audio.

Arguments:
1. dataset: Path to the folder containg the .wav file of the maestro or fma dataset.
1. meta: Path to the metadata of the dataset
1. name: Choose between 'fma' or 'maestro'
1. target: folder in which the newly created tfrecord files will be saved
1. \[optional\] sr: Each audio can be resampled to sr 
1. \[optional\] duration: Each audio can be split into chunk of *duration* seconds

## Train the network
To train the dataset, use the script [train](python/train.py)
This create will restore the last checkpoint, train the model and save a checkpoint at the end of each epoch. 
All argument have coherent default value. 

Arguments:
1. ckpt: path to the folder containing the model checkpoint. Default to ../ckpt/*model_name*/*length*/ 
1. model: Choose between 'igan' or 'pgan'
1. train: Path to the training tfrecord file. Default to '../fma_dataset/train.tfrecord'
1. val: Path to the validation tfrecord file. Default to '../fma_dataset/val.tfrecord'
1. test: Path to the test tfrecord file.  Default to '../fma_dataset/test.tfrecord'
1. epoch: Number of epoch to train the network. Default is 100.
1. log: Path to the folder containg the training log. default to ../log/*model_name*/*length*/ 
1. size: Length of the frame to inpaint/predict in second. Default to 0.064.
1. batch: Batch size. default to 256

## Evaluate the network
Three scripts can be used to evaluate the network: [evaluate](python/evaluate.py), [PerceptualEval](python/PerceptualEval.py) and [analyze_folder](matlab/analyze_folder.m)
The first one output the SNR on spectrum, audio and can also plot the waveform and spectrums.
The second one generate audio sample of 2s that fit an evaluation with the ODG.
The last one compute the ODG

### Script Evaluate
This script computes the SNR on audio and spectrum and can optionally plot the waveform and spectrum.

Arguments:

1. model: Either pgan for prediction or igan for inpainting.
1. ckpt: Path to the folder containing the model checkpoint. Default to ../ckpt/*model_name*/*length*/ 
1. data: Path to the testing tfrecord file. Default to  '../fma_dataset/test.tfrecord'
1. plot: Whether to plot the spectrum and waveform. Choose between 'plot' and 'noplot'
1. target: Path to the folder that will contains the newly created waveform and spectrum. If plot='noplot', no file will be created
1. length: Length of the frame to inpaint/predict in second. Default to 0.064.
1. batch: Batch size. default to 256

### Script PerceptualEval
This script creates sample of 2s long either by filling a gap at 0.5s or by predicting the whole audio by chunk of *length* seconds. The conditioning frame are taken from the original.

Arguments:

1. model: Either pgan for prediction or igan for inpainting.
1. ckpt: Path to the folder containing the model checkpoint. Default to ../ckpt/*model_name*/*length*/ 
1. data: Path to the testing tfrecord file. Default to  '../fma_dataset/test.tfrecord'
1. target: Path to the folder that will contains the newly created waveform and spectrum. If plot='noplot', no file will be created. Default to 'noplot'
1. method: Either 'single_hole' or chaining. Single hole will fill in a gap at 0.5s. Chaining will predict 2s of audio by chunk of *length* seconds.
1. length: Length of the frame to inpaint/predict in second. Default to 0.064.

### Script analyze_folder
This script computes the average ODG in a folder. 
Each reference signal must be named 'or_*X*.wav'.
Each signal under test must be named 'rec_*Y*.wav'
For each reference signal name 'or_*X*.wav' the folder must contains the signal under test in a file 'rec_*Y*.wav' with *X*=*Y* and *X*, *Y* are string

Change the variable *audio_folder* to the folder containing all the reference and signal under test.



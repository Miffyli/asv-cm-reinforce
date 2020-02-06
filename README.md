# ASV and CM optimization with REINFORCE

Speaker verification and spoofing countermeasure systems are trained independent, but used together.
What if we optimize them together for the tandem task?

This repository contains the code for replicating experiments in [TODO paper link] 

## Requirements
* PyTorch (experiments ran on v1.3.1)
* [Kaldi](https://kaldi-asr.org/) and [PyKaldi](https://github.com/pykaldi/pykaldi) for extracting features
* [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [ASVSpoof19](https://datashare.is.ed.ac.uk/handle/10283/3336) (logical access)  datasets

## Preprocessing and feature extraction

Before experiments we need the features used, split into two different parts.

### Extracting x-vectors

x-vectors for ASV are extracted with Kaldi's pretrained model. See `kaldi_xvector_extraction`
directory and the README within for how to extract x-vectors for bunch of wav files. 
Note that you need to extract these features for VoxCeleb1 and ASVSpoo19 lists.

See `utils/audio_files_to_wav16.py` for converting different audio files to 16kHZ .wav files
for this extraction.

After extracting x-vectors to Kaldi's format, use `utils/kaldi_to_numpy.py` to convert the created
Kaldi files into Numpy .mat files, one per utterance. Save the features under following structure:

```
features/xvectors/ASVspoof2019_LA_train/wav/{LA_T_1000137.npy, LA_T_1000406.npy, ...}
features/xvectors/ASVspoof2019_LA_dev/wav/{LA_D_1000265.npy, LA_D_1000752.npy, ...}
features/xvectors/ASVspoof2019_LA_eval/wav/{LA_E_1000147.npy, LA_E_1000273.npy, ...}
features/xvectors/VoxCeleb/wav/{A.J._Buckley/, A.R._Rahman/, ...}
```

### Extracting CQCC features

You only have to extract CQCC features for ASVSpoof19 data. 

Follow the instructions in `cqcc_extraction` directory, then use `utils/mats_to_numpy.py` to convert
the `.mat` files into `.npy` files. The key used in `.mat` files is "CQcc".

Place the resulting CQCC features under `features` directory like so:

```
features/cqcc/ASVspoof2019_LA_train/wav/{LA_T_1000137.npy, LA_T_1000406.npy, ...}
features/cqcc/ASVspoof2019_LA_dev/wav/{LA_D_1000265.npy, LA_D_1000752.npy, ...}
features/cqcc/ASVspoof2019_LA_eval/wav/{LA_E_1000147.npy, LA_E_1000273.npy, ...}
```

### Gather filelists

Copy ASVSpoof19 protocol filelists under `lists/` directory (e.g. "ASVspoof2019.LA.cm.train.trn", "ASVspoof2019.LA.asv.eval.male.trn.txt").

Use `utils/split_voxceleb_train.py` to create a filelist for training ASV, name it "VoxCeleb_asv_train_list.txt" and place it under `lists/`.


## Running experiments

After preprocessing command `./scripts/run_all.sh` in the root directory to run all experiments.

If scripts run without errors, you should have `output` directory with different text files, images
and videos. Main ones of these are

* `initial_eval.txt`, which contains the t-DCF, ASV EER and CM EER of the initial (pretrained) models
* `joint_training_eval.txt`, which contains the final t-DCF, ASV EER and CM EER of joint training with different methods.
   `ce_same` refers to "independent models, same labels" and `ce_split` is "independent models, split labels".
* `*averages_relative.pdf` contain the learning curves of different methods, *averaged over all repetitions*. For individual
   curves per repetition, see `*_metrics.png`
* `mp4` videos contain evolution of scores and DET curves over training.


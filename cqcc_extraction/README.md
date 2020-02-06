# Extracting CQCC features with Octave

Code here is copy of CQCC baseline code provided
along with with ASVSpoof19 challenge, available [here](https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip),
but modified to work with Octave. The resulting CQCC features
differ from ones extracted with MATLAB, but in the CQCC baseline method
the performance did not change.

How to extract CQCC features for .wav files
1) Create a file listing all wav files you want to extract CQCCs for (one line per wav file). See `utils/wavs_to_list.py`.
2) Launch Octave and call `extract_based_on_list` with path to the filelist mentioned above.

This should create bunch of `.mat` files beside the original `.wav` files, each containing
the CQCC features of the respective WAV file.



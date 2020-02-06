# Extracting x-vectors with kaldi

Tools under this directory are meant to be placed under
"[kaldi_dir]/egs/voxceleb/v2", which also should include a
pretrained model from kaldi websites (trained on voxceleb):
https://kaldi-asr.org/models/m7

These tools extract individual x-vectors for a list of .wav
files:
1) Create a file listing all wav files you want to extract x-vectors for (one line per wav file). See `utils/wavs_to_list.py`.
2) Update variables in extract_xvectors_main.sh to point at the correct folders/files
3) Run `extract_xvectors_main.sh`
4) If things go correctly, you should have spk_xvectors.scp etc files under the directories somewhere, which contain the extracted xvectors (one per .wav file. Speaker-id is set to be the path to that wav file)


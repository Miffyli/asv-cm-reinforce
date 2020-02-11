## Precomputed scores

This directory contains some of the scores used in the paper "[An initial investigation on optimizing tandem speaker verification and countermeasure systems using reinforcement learning](https://arxiv.org/abs/2002.03801)", compressed into a `.zip` file.

Specifically, this contains scores ran on ASVSpoof19 `ASVspoof2019.LA.asv.{subset}.gi.trl` lists, separated by the subset (dev, eval) and ASV and CM system. There are three repetitions of each experiement, following files from each:

* `siamese_asv` or `siamese_cm`: The initial systems after pretraining
* `pg_simple`: After "REINFORCE Simple" training
* `pg_reward`: After "REINFORCE Reward" training
* `pg_penalize`: After "REINFORCE Penalize" training
* `pg_tdcf`: After "REINFORCE t-DCF" training
* `ce`: After "independent models, same labels" training
* `ce_split`: After "independent models, split labels" training

To save space, this archive only contains the score-files at the end of the tandem training, not the intermediate scores used to create figures
in the paper.

### Score-file structure

Rows in each of the score files have following structure:

```
[is_target] [score] [spoof_system]
```

* `is_target` is either `True` or `False`, indicating if the corresponding trial should be passed by tandem ASV+CM system (`True` only if trial is target speaker _and_ bonafide). 
* `score` is the score from the respective system. Higher value supports accepting.
* `bonafide` is either `bonafide` or `A0#`. If `bonafide`, trial has a bona fide (non-spoof) sample. If one of `A0#`, then this column tells by which system the spoof sample was generated (see [ASVSpoof19](https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/asvspoof2019_Interspeech2019_submission.pdf?sequence=2&isAllowed=y) for more info on these).

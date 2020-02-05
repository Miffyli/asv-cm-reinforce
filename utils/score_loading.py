# Utilities for reading score-files
import numpy as np

from utils.data_loading import readlines_and_split_spaces

def load_scorefile_and_split_to_arrays(scorefile_path):
    """
    Load a scorefile where each line has multiple columns
    separated by whitespace, and split each column to its own
    array
    """
    scorefile_lines = readlines_and_split_spaces(scorefile_path)

    arrays = [np.array(column) for column in zip(*scorefile_lines)]

    return arrays


def load_scorefile_and_split_scores(scorefile_path):
    """
    Load a scorefile with following structure and
    return three arrays: target_scores, nontarget_scores and original_scores

    Each line is
       is_target score [optional ...]

    where is_target is either "True" or "False".
    score is a float
    """
    scorefile_lines = readlines_and_split_spaces(scorefile_path)

    target_scores = []
    nontarget_scores = []
    original_scores = []

    for score_line in scorefile_lines:
        # Some trials are None (because files are missing).
        # Skip them
        if score_line[1] == "None":
            continue

        is_target = score_line[0] == "True"
        score = float(score_line[1])
        original_scores.append(score)

        if is_target:
            target_scores.append(score)
        else:
            nontarget_scores.append(score)

    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)
    original_scores = np.array(original_scores)

    return target_scores, nontarget_scores, original_scores

import numpy as np
import json
import sys
import os
from pathlib import Path
from collections import OrderedDict

# Load the dictionary of label sequences
with open(Path(__file__).parent / 'task_sequences.json') as f:
    TASK_SEQUENCES = {k: OrderedDict(sorted(v.items(), key=lambda x: x[1])) for k, v in json.load(f).items()}

class LabelMapper:
    # special cases of label values
    UNCERTAIN = -1
    MISSING = -2

    def __init__(self, from_seq, to_seq):
        """Class that converts one task sequence,
        to another task sequence. (e.g nih to stanford).

        The key equation is: x_new = Ax + b where
        A is the mapping_matrix, putting 1s in x to the
        right place in x_new. b, below known as missing_bias
        makes ure that the values in the to_seq that don't exist
        in the from_seq, are all put to eqaul zero.

        Args:
            from_seq: An ordered dict of the tasks (task: index)
               you want to map from.

            to_seq: An ordered dict of the tasks (task: index)
               you want to map to.
        """
        # Can't be any duplicates within a task sequence.
        assert len(set(from_seq)) == len(from_seq)
        assert len(set(to_seq)) == len(to_seq)

        # The values 0 .. num_pathologies need to be unique
        assert len(set(to_seq.values())) == len(to_seq.values())
        assert len(set(from_seq.values())) == len(from_seq.values())

        # store the from and to task sequences
        self.from_seq = from_seq
        self.to_seq = to_seq

        # create the mapping matrix
        self.mapping_matrix = self._get_map_matrix(from_seq, to_seq)

        # Each row in the mapping matrix that is all zero
        # corresponds to a task that does not exist in the from_seq
        # we want those values to have value -2
        # These values can then easily be masked at a later stage

        missing_tasks_indeces = np.where(np.sum(self.mapping_matrix, axis=1) == 0)

        self.missing_bias = np.zeros(len(to_seq))
        self.missing_bias[missing_tasks_indeces] = LabelMapper.MISSING

    def map(self, label):
        """Maps label from self.from_seq to self.from_to_seq

            The missing_bias makes sure that tasks that are missing
            in the from_seq are put as -2 in new_label.

        Args:
            label: A numpy array (a vector) with binary values.
            each corresponding to a binary task. Usually this task is
            to determine if whether specific pathology is present.

        Return:
            new_label: A numpy array with the labels whose indeces corresponds
            to the label sequence stored in self.to_seq.
        """

        new_label = np.dot(self.mapping_matrix, label) + self.missing_bias

        return new_label

    def _get_map_matrix(self, from_seq, to_seq):
        """ Creates a mapping matrix between to
        labeling sequences.

        The matrix shape is (num_from_tasks, num_to_tasks).
        That means that if a row ends up fully empty, that class
        does not exists in the from_seq. If a column ends up fully
        empty it means that the class does not exist in the target.
        """
        num_from_tasks = len(from_seq)
        num_to_tasks = len(to_seq)
        map_matrix = np.zeros((num_to_tasks, num_from_tasks))

        for target_pathology in to_seq:
            to_id = to_seq[target_pathology]
            if target_pathology in from_seq:
                from_id = from_seq[target_pathology]
                map_matrix[to_id, from_id] = 1

        return map_matrix

    def label_overlap(self):
        """Utility method to check overlap
        between the two label_sequences"""

        overlap = set(self.from_seq).intersection(set(self.to_seq))

        return list(overlap)

    @staticmethod
    def display(sequence, array):
        """Prints in easy to read format the binary array
           and label sequence.

        Put this in this class mainly for namespacing purposes.
        """

        tasks = list(sequence)
        array = array.tolist()
        assert(len(tasks) == len(array))

        path_label_dict = dict(zip(tasks, array))

        print(json.dumps(path_label_dict, indent=4))

        return dict(zip(tasks, array))




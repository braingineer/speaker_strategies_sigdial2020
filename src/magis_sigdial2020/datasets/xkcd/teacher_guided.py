from magis_sigdial2020.datasets.xkcd.vectorized import XKCD
from magis_sigdial2020.utils.data import Dataset
import numpy as np



class TeacherGuidedXKCD(Dataset):
    def __init__(self, teacher_phi_path, xkcd_coordinate_system='x-y'):
        self.xkcd = XKCD.from_settings(coordinate_system=xkcd_coordinate_system)
        self.n_colors = len(self.xkcd.color_vocab)
        self.teacher_phi = np.load(teacher_phi_path).astype(np.float32)
        self.split = None
        self.set_split("train")
        self._teacher_phi_path = teacher_phi_path

    def get_teacher_phi_path(self):
        return self._teacher_phi_path
    
    def set_split(self, split):
        self.xkcd.set_split(split)
        self.split = split

    def __getitem__(self, index):
        output = self.xkcd[index]
        if self.split == "train":
            teacher_phi = self.teacher_phi[index]
        else:
            teacher_phi = np.zeros(self.n_colors).astype(np.float32)
            teacher_phi[output['y_color_name']] = 1
        output['teacher_phi'] = teacher_phi
        return output

    def __len__(self):
        return len(self.xkcd)
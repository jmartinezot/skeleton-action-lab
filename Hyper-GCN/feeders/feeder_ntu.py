import os
import numpy as np

from torch.utils.data import Dataset

from feeders import tools
from feeders.tools import align


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, num_clips=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, frame_sample='resize', align=False, spatial_flip=False, drop_joint=False,
                 drop_axis=False, window_size=-1, normalization=False, debug=False, use_mmap=False, bone=False, vel=False):
        """
        :param data_path: Path to data (.npz from Kaggle with x_train/x_test or CTR-style .npy).
        :param label_path: Optional label file for non-Kaggle layouts.
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_clips = num_clips
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.align = align
        self.frame_sample = frame_sample
        self.spatial_flip = spatial_flip
        self.drop_joint = drop_joint
        self.drop_axis = drop_axis
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    @staticmethod
    def _convert_labels(raw_label):
        if raw_label.ndim == 2:
            return np.where(raw_label > 0)[1]
        if raw_label.ndim == 1:
            return raw_label
        raise ValueError(f'Unsupported label shape: {raw_label.shape}')

    @staticmethod
    def _reshape_kaggle(x):
        if x.ndim != 3:
            raise ValueError(f'Expected Kaggle data with shape (N, T, 150); got {x.shape}')
        N, T, D = x.shape
        if D != 150:
            raise ValueError(f'Expected Kaggle D=150 (25*3*2); got D={D}')
        # (N, T, 2, 25, 3) -> (N, 3, T, 25, 2)
        return x.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2).astype(np.float32, copy=False)

    @staticmethod
    def _reshape_ctrgcn(x):
        if x.ndim != 5:
            raise ValueError(f'Expected CTR-GCN tensor shaped (N, C, T, V, M); got {x.shape}')
        return x.astype(np.float32, copy=False)

    def load_data(self):
        # data: N C V T M
        mmap_mode = 'r' if self.use_mmap else None
        raw = np.load(self.data_path, allow_pickle=True, mmap_mode=mmap_mode)

        if isinstance(raw, np.lib.npyio.NpzFile) and f'x_{self.split}' in raw and f'y_{self.split}' in raw:
            data = raw[f'x_{self.split}']
            label = raw[f'y_{self.split}']
            data = self._reshape_kaggle(data)
            label = self._convert_labels(label)
            split_prefix = self.split
        else:
            if self.label_path is None:
                raise ValueError('label_path must be provided for non-Kaggle data layouts.')
            data = self._reshape_ctrgcn(raw)
            label_raw = np.load(self.label_path, allow_pickle=True)
            label = self._convert_labels(label_raw)
            split_prefix = os.path.splitext(os.path.basename(self.data_path))[0]

        if self.debug:
            data = data[:100]
            label = label[:100]

        self.data = data
        self.label = label
        self.sample_name = [f'{split_prefix}_{i}' for i in range(len(self.data))]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        if self.frame_sample == 'clips':
            data_numpy = tools.get_clips(data_numpy[:, :valid_frame_num], self.window_size)
        elif self.frame_sample == 'resize':
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        else:
            raise ValueError('frame_sample only supports clips or resize')
        if self.align:
            data_numpy = tools.align(data_numpy)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            from .bone_pairs import ntu_pairs
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            bone_data_numpy[:, :, 20] = data_numpy[:, :, 20]
            data_numpy = bone_data_numpy
        elif not self.vel:
            trajectory = data_numpy[:, :, 20]
            data_numpy = data_numpy - data_numpy[:, :, 20:21]
            data_numpy[:, :, 20] = trajectory

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


import numpy as np

from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
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
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: (N, T, D=150) Kaggle one-hot format OR (N, C, T, V, M) CTR-GCN layout
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.split == 'train':
            raw_data = npz_data['x_train']
            raw_label = npz_data['y_train']
            self.sample_name = ['train_' + str(i) for i in range(len(raw_data))]
        elif self.split == 'test':
            raw_data = npz_data['x_test']
            raw_label = npz_data['y_test']
            self.sample_name = ['test_' + str(i) for i in range(len(raw_data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        # Labels can be one-hot or already integer class ids
        self.label = self._extract_labels(raw_label)
        # Data can be flat Kaggle layout or already 5D CTR-GCN layout
        self.data = self._reshape_data(raw_data)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    @staticmethod
    def _extract_labels(raw_label):
        label = np.asarray(raw_label)
        if label.ndim == 1:
            return label.astype(np.int64)
        if label.ndim == 2:
            # One-hot or 2D class matrix
            return label.argmax(axis=1).astype(np.int64)
        raise ValueError(f"Unsupported label shape {label.shape}; expected 1D class ids or 2D one-hot.")

    @staticmethod
    def _reshape_data(data):
        arr = np.asarray(data)
        if arr.ndim == 5:
            # Assume already (N, C, T, V, M)
            return arr
        if arr.ndim != 3:
            raise ValueError(f"Unsupported data shape {arr.shape}; expected 3D flat or 5D tensor.")

        n, t, d = arr.shape
        if d not in (150, 75):
            raise ValueError(f"Unexpected feature dimension D={d}; expected 150 (25*3*2) or 75 (25*3*1).")

        v = 25
        c = 3
        m = 2 if d == 150 else 1
        arr = arr.reshape((n, t, m, v, c)).transpose(0, 4, 1, 3, 2)
        return arr

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
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
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

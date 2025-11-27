import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class KaggleFeeder(Dataset):
    """
    Dataset that reads the Kaggle NTU60 one-hot NPZ directly.
    Expects keys x_train/y_train and x_test/y_test shaped (N, T, 150).
    Outputs tensors shaped (N, C=3, T, V=25, M=2) with integer labels.
    """

    def __init__(
        self,
        data_path,
        split="train",
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=False,
    ):
        self.data_path = data_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.debug = debug
        # NPZ cannot be memory-mapped; keep the flag for interface parity.
        self.use_mmap = use_mmap

        self.load_data()
        if normalization:
            self.get_mean_map()

    @staticmethod
    def _reshape_x(x):
        """
        (N, T, 150) -> (N, C=3, T, V=25, M=2)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x to have 3 dims (N, T, 150); got {x.shape}")
        n, t, d = x.shape
        if d != 150:
            raise ValueError(f"Expected D=150 (25*3*2), got D={d}")

        x = x.reshape(n, t, 25, 3, 2)
        x = np.transpose(x, (0, 3, 1, 2, 4))
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _convert_labels(y):
        """
        Convert one-hot labels to class indices if needed.
        """
        if y.ndim == 2:
            return y.argmax(axis=1).astype(np.int64, copy=False)
        if y.ndim == 1:
            return y.astype(np.int64, copy=False)
        raise ValueError(f"Unsupported label shape: {y.shape}")

    def load_data(self):
        data = np.load(self.data_path, allow_pickle=True)
        x_key = f"x_{self.split}"
        y_key = f"y_{self.split}"
        if x_key not in data.files or y_key not in data.files:
            raise KeyError(
                f"Missing keys '{x_key}'/'{y_key}' in NPZ. Found: {data.files}"
            )

        x = data[x_key]
        y = data[y_key]

        x = self._reshape_x(x)
        y = self._convert_labels(y)

        if self.debug:
            x = x[:100]
            y = y[:100]

        self.data = x
        self.label = y
        self.sample_name = [f"{self.split}_{i:06d}" for i in range(len(y))]

    def get_mean_map(self):
        data = self.data
        n, c, t, v, m = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((n * t * m, c * v)).std(axis=0).reshape((c, 1, v, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = int(self.label[index])

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

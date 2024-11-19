import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, few_shot_data_file=None, few_shot_data_file_run=None, label_path=None, p_interval=1, split='train', 
                 random_choose=False, random_shift=False, random_move=False, random_rot=False, window_size=-1, 
                 normalization=False, debug=False, use_mmap=False, bone=False, vel=False, IL_step=0, k_shot=-1,
                 labels_prev_step=0, train_label_ids=None, eval_label_ids=None, label_order_protocol=0):
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
        self.standard_data_path = data_path
        self.fewshot_data_path = few_shot_data_file_run
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
        self.IL_step = IL_step
        self.k_shot = k_shot
        self.labels_prev_step = labels_prev_step

        print(f'\nCHECK: few shot data file {self.fewshot_data_path}, standard data file {self.standard_data_path}')

        base_map = {i: i for i in range(40)}

        label_ordering = {
            0: {40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49,
                50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59},
            1: {40: 50, 41: 51, 42: 52, 43: 53, 44: 54, 45: 55, 46: 56, 47: 57, 48: 58, 49: 59,
                50: 40, 51: 41, 52: 42, 53: 43, 54: 44, 55: 45, 56: 46, 57: 47, 58: 48, 59: 49},
            2: {40: 45, 41: 46, 42: 47, 43: 48, 44: 49, 45: 40, 46: 41, 47: 42, 48: 43, 49: 44,
                50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59},
            3: {40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49,
                50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44}
        }

        reverse_ordering = {
            0: {40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49,
                50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59},
            1: {40: 50, 41: 51, 42: 52, 43: 53, 44: 54, 45: 55, 46: 56, 47: 57, 48: 58, 49: 59,
                50: 40, 51: 41, 52: 42, 53: 43, 54: 44, 55: 45, 56: 46, 57: 47, 58: 48, 59: 49},
            2: {40: 45, 41: 46, 42: 47, 43: 48, 44: 49, 45: 40, 46: 41, 47: 42, 48: 43, 49: 44,
                50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59},
            3: {40: 55, 41: 56, 42: 57, 43: 58, 44: 59, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49,
                50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 40, 56: 41, 57: 42, 58: 43, 59: 44}
        }

        self.forward_map = {**base_map, **label_ordering[label_order_protocol]}
        self.reverse_map = {**base_map, **reverse_ordering[label_order_protocol]}

        self.train_label_ids = [self.forward_map[i] for i in train_label_ids]
        self.eval_label_ids = []
        for i in eval_label_ids:
            if i >= 40:
                self.eval_label_ids.append(self.forward_map[i])
            else:
                self.eval_label_ids.append(i)

        print(f'loading train ids: {self.train_label_ids}, loading eval ids: {self.eval_label_ids}')

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.k_shot > 0:
            npz_data = np.load(self.fewshot_data_path)
        else:
            import pdb
            pdb.set_trace()
            npz_data = np.load(self.standard_data_path)

        if self.split == 'train':
            self.all_data = npz_data['x_train']
            self.all_label = np.where(npz_data['y_train'] > 0)[1]
            self.all_sample_name = ['train_' + str(i) for i in range(len(self.all_data))]
            self.select_ids = self.train_label_ids

        elif self.split == 'test':
            self.all_data = npz_data['x_test']
            self.all_label = np.where(npz_data['y_test'] > 0)[1]
            self.all_sample_name = ['test_' + str(i) for i in range(len(self.all_data))]
            self.select_ids = self.eval_label_ids
        else:
            raise NotImplementedError('data split only supports train/test')

        # select only relevant label indices for the CL step
        selected_data = []
        selected_label = []
        selected_sample_name = []
        actual_selected_loaded_label = []

        for ind, i in enumerate(self.all_label):
            if i in self.select_ids:
                selected_data.append(self.all_data[ind])
                actual_selected_loaded_label.append(i)
                # modify here, re-map for sequantial training
                if self.split == 'train' or (self.split == 'test' and i >= 40):
                    i = self.reverse_map[i]
                selected_label.append(i)
                selected_sample_name.append(self.all_sample_name[ind])

        self.data = np.stack(selected_data, 0)
        self.label = np.array(selected_label)
        self.sample_name = selected_sample_name

        print(f'\nsplit {self.split}, loaded data {self.data.shape}, labels {self.label.shape}')
        # print(f'\nactual data loaded, (unmapped) label ids: {np.unique(np.array(actual_selected_loaded_label))}')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

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

import logging
from collections import defaultdict
from math import ceil
from os import listdir
from os.path import join, splitext, exists

import cv2 as cv
import mnist
import numpy as np
import torch
from albumentations import OpticalDistortion, GridDistortion, ElasticTransform
from torch.nn import functional as F

from augmentation import RandomSizedCropAlbuAug, GaussianBlurAug, MotionBlurAug, AlbuAug, PickOne, FlipAug, ApplyAll
from model import CNNPreProcessor, SequencePredictor, build_basic_encoder, build_basic_decoder
from utils import NoiseType, add_noise, put_side_by_side, opencv_show, load_json

logger = logging.getLogger(__name__)


def one_hot_encode(index, num_max, leave_one_out=True):
    res = np.array([int(i == index) for i in range(num_max)])
    if leave_one_out:
        return res[1:]
    return res


def get_augmentations():
    crop_p = 0.4
    blur_p = 0.4
    distort1_p = 0.2
    distort2_p = 0.4
    flip_p = 0.4

    crop = RandomSizedCropAlbuAug(crop_p)

    gauss_blur = GaussianBlurAug(p=blur_p, kernel_sizes=range(7, 16, 2))
    motion_blur = MotionBlurAug(p=blur_p, kernel_sizes=(3, 5))

    optical_distort = AlbuAug(OpticalDistortion(p=distort1_p, distort_limit=1, shift_limit=0.5))
    grid_distort = AlbuAug(GridDistortion(p=distort1_p))
    elastic1 = AlbuAug(ElasticTransform(p=distort2_p, alpha=40, sigma=90 * 0.05, alpha_affine=90 * 0.05))
    elastic2 = AlbuAug(ElasticTransform(p=distort1_p))
    blur_aug = PickOne([gauss_blur, motion_blur])
    distort_aug = PickOne([optical_distort, grid_distort, elastic1, elastic2])
    flip_aug = FlipAug(p=flip_p)

    return ApplyAll([crop, blur_aug, distort_aug, flip_aug])


class BasicDataset(object):

    def __init__(self, data_paths, batch_size, preprocessor, augmentations=None, include_ids=None, shuffle=True,
                 prepare_on_load=True):
        self.debug = False
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.prepare_on_load = prepare_on_load
        self.data_d = {}
        self.load_data(include_ids)

    @property
    def size(self) -> int:
        return len(self.data_d.keys())

    @property
    def batches_per_epoch(self) -> int:
        return ceil(self.size / self.batch_size)

    def load_data(self, include_ids=None):
        if include_ids is not None:
            include_ids = set(include_ids)

        for p in self.data_paths:
            for fn in listdir(p):
                if include_ids is not None and fn not in include_ids:
                    continue
                full_fn = join(p, fn)

                if full_fn in self.data_d:
                    logger.warning(f"Duplicate file name {fn}")
                self.data_d[full_fn] = self.preprocessor.prepare_im(
                    cv.imread(full_fn)) if self.prepare_on_load else cv.imread(full_fn)

    def batches(self):
        raise NotImplementedError()


class SequenceDataset(BasicDataset):

    def __init__(self, data_paths, batch_size, preprocessor, augmentations=None, include_ids=None, shuffle=True,
                 prepare_on_load=True, num_actions=0, seq_length=10, leave_one_out_encoding=True):
        super().__init__(data_paths, batch_size, preprocessor, augmentations, include_ids, shuffle, prepare_on_load)
        if seq_length < 2:
            raise ValueError("Need sequence length of at least 2")
        self.with_actions = num_actions > 0
        self.num_actions = num_actions
        self.seq_length = seq_length
        self.leave_one_out = leave_one_out_encoding
        self.sequences = []
        self.generate_sequences()

    @property
    def size(self) -> int:
        return len(self.sequences)

    def generate_sequences(self):
        self.sequences = []

        for seq_id, lst in self.data_d.items():
            if len(lst) < self.seq_length:
                logger.warning(f"Sequence {seq_id} is too short for sequence length {self.seq_length}")
                continue

            lst_idx = 0
            while lst_idx + self.seq_length <= len(lst):
                seq = lst[lst_idx: lst_idx + self.seq_length]
                self.sequences.append(seq)
                lst_idx += 1

        logger.info(f"Generated {len(self.sequences)} sequences of length {self.seq_length}")

    def load_data(self, include_ids=None):
        self.data_d = defaultdict(list)
        if isinstance(include_ids, dict) and {type(v) for v in include_ids.values()} == {list}:
            for p in self.data_paths:
                for seq_id, lst in include_ids.items():
                    for obj in lst:
                        action_id = obj.get("action")
                        seq_idx = obj["seq_idx"]
                        fn = obj["fn"]

                        full_fn = join(p, fn)

                        if not exists(full_fn):
                            continue

                        im = self.preprocessor.prepare_im(cv.imread(full_fn)) if self.prepare_on_load else cv.imread(
                            full_fn)
                        self.data_d[seq_id].append({"im": im, "seq_idx": seq_idx, "action": action_id})
        else:
            for p in self.data_paths:
                for fn in listdir(p):
                    if include_ids is not None and fn not in include_ids:
                        continue
                    full_fn = join(p, fn)
                    no_ending_split = splitext(fn)[0].split("_")
                    action_id = None
                    seq_idx = int(no_ending_split[-1])
                    if self.with_actions:
                        action_id = int(no_ending_split[-2])
                        seq_id = "_".join(no_ending_split[:-2])
                    else:
                        seq_id = "_".join(no_ending_split[:-1])

                    im = self.preprocessor.prepare_im(cv.imread(full_fn)) if self.prepare_on_load else cv.imread(
                        full_fn)

                    obj = {"im": im, "seq_idx": seq_idx, "action": action_id}
                    self.data_d[seq_id].append(obj)

        for k in sorted(self.data_d.keys()):
            self.data_d[k] = sorted(self.data_d[k], key=lambda x: x["seq_idx"])

    def actions_to_tensor(self, seq):
        a_seq = [one_hot_encode(el, self.num_actions, leave_one_out=self.leave_one_out) for el in seq]
        return torch.cat([torch.from_numpy(el).unsqueeze(0).unsqueeze(0).type(torch.float32) for el in a_seq])

    def batches(self):
        raise NotImplementedError()


class SequenceReconstructionDataset(SequenceDataset):

    def __init__(self, data_paths, batch_size, preprocessor, augmentations=None, include_ids=None, shuffle=True,
                 prepare_on_load=True, num_actions=0, seq_length=10, leave_one_out_encoding=True):
        super().__init__(data_paths, batch_size, preprocessor, augmentations, include_ids, shuffle, prepare_on_load,
                         num_actions, seq_length, leave_one_out_encoding)

    def batches(self):
        data = self.sequences

        if self.shuffle:
            data = [data[i] for i in np.random.permutation(len(data))]

        curr_idx = 0
        while curr_idx < len(data):
            x_batch_list = []
            a_batch_list = []

            while curr_idx < len(data) and len(x_batch_list) < self.batch_size:
                seq_ims = [el["im"].copy() for el in data[curr_idx]]

                if self.debug:
                    opencv_show(*seq_ims, prefix="Before Augmentation ")

                if self.augmentations is not None:
                    # TODO implement
                    pass

                if self.debug:
                    opencv_show(*seq_ims, prefix="After Augmentation ")

                x_tensor = self.preprocessor.preprocess_im_sequence(seq_ims)

                if self.num_actions > 0:
                    a_batch_list.append(self.actions_to_tensor([e["action"] for e in data[curr_idx][:-1]]))

                x_batch_list.append(x_tensor)

                curr_idx += 1

            x_batch = torch.cat(x_batch_list, dim=1)
            a_batch = None if len(a_batch_list) == 0 else torch.cat(a_batch_list, dim=1)

            yield x_batch, a_batch


class SequencePredictionDataset(SequenceDataset):

    def __init__(self, data_paths, batch_size, preprocessor, augmentations=None, include_ids=None, shuffle=True,
                 prepare_on_load=True, num_actions=0, seq_length=10, leave_one_out_encoding=True):
        super().__init__(data_paths, batch_size, preprocessor, augmentations, include_ids, shuffle, prepare_on_load,
                         num_actions, seq_length, leave_one_out_encoding)

    def batches(self):
        data = self.sequences

        if self.shuffle:
            data = [data[i] for i in np.random.permutation(len(data))]

        curr_idx = 0
        while curr_idx < len(data):
            x_batch_list = []
            a_batch_list = []
            y_batch_list = []

            while curr_idx < len(data) and len(x_batch_list) < self.batch_size:
                seq_ims = [el["im"].copy() for el in data[curr_idx]]
                in_seq = seq_ims[:-1]
                out_im = seq_ims[-1]

                if self.debug:
                    opencv_show(*seq_ims, prefix="Before Augmentation ")

                if self.augmentations is not None:
                    # TODO implement
                    pass

                if self.debug:
                    opencv_show(*seq_ims, prefix="After Augmentation ")

                x_tensor = self.preprocessor.preprocess_im_sequence(in_seq)
                y_tensor = self.preprocessor.preprocess(out_im)

                if self.num_actions > 0:
                    a_batch_list.append(self.actions_to_tensor([e["action"] for e in data[curr_idx][:-1]]))

                x_batch_list.append(x_tensor)
                y_batch_list.append(y_tensor)

                curr_idx += 1

            x_batch = torch.cat(x_batch_list, dim=1)
            a_batch = None if len(a_batch_list) == 0 else torch.cat(a_batch_list, dim=1)
            y_batch = torch.cat(y_batch_list)

            yield x_batch, a_batch, y_batch


class AutoEncoderDataset(BasicDataset):

    def __init__(self, data_paths, batch_size, preprocessor, augmentations=None, include_ids=None, shuffle=True,
                 prepare_on_load=True):
        super().__init__(data_paths, batch_size, preprocessor, augmentations=augmentations, include_ids=include_ids,
                         shuffle=shuffle, prepare_on_load=prepare_on_load)

    @classmethod
    def mnist_train_val(cls, batch_size, target_shape, augmentations=None, shuffle=True, val_ratio=0.1,
                        split_seed=None):
        ims = mnist.train_images()
        if split_seed is not None:
            rs = np.random.RandomState(split_seed)
            ims_shuffled = [ims[i] for i in rs.permutation(len(ims))]
        else:
            ims_shuffled = [ims[i] for i in np.random.permutation(len(ims))]

        train_ims = ims_shuffled[int(val_ratio * len(ims_shuffled)):]
        val_ims = ims_shuffled[:int(val_ratio * len(ims_shuffled))]

        preprocessor = CNNPreProcessor(bgr_mean=0.1307, bgr_std=0.3081, target_shape=target_shape)
        train_ds = cls([], batch_size, preprocessor, augmentations=augmentations, shuffle=shuffle)
        for idx, im in enumerate(train_ims):
            train_ds.data_d[f"{idx}"] = im
        val_ds = cls([], batch_size, preprocessor, shuffle=shuffle)
        for idx, im in enumerate(val_ims):
            val_ds.data_d[f"{idx}"] = im
        return train_ds, val_ds

    def batches(self):
        data = [v for k, v in sorted(self.data_d.items())]

        if self.shuffle:
            data = [data[i] for i in np.random.permutation(len(data))]

        curr_idx = 0
        while curr_idx < len(data):
            batch_list = []

            while curr_idx < len(data) and len(batch_list) < self.batch_size:
                sample_im = data[curr_idx].copy()

                if self.debug:
                    cv.imshow("Before Augmentation", sample_im)
                    cv.waitKey()
                    cv.destroyAllWindows()

                if self.augmentations is not None:
                    sample_im, _ = self.augmentations.augment(sample_im, np.zeros_like(sample_im))

                if self.debug:
                    cv.imshow("After Augmentation", sample_im)
                    cv.waitKey()
                    cv.destroyAllWindows()

                x_tensor = self.preprocessor.preprocess(sample_im)
                batch_list.append(x_tensor)

                curr_idx += 1

            yield torch.cat(batch_list)


class DeNoisingAutoEncoderDataset(AutoEncoderDataset):

    def __init__(self, data_paths, batch_size, preprocessor, augmentations=None, include_ids=None, shuffle=True,
                 noise_generator=None, noise_probability=0.5, prepare_on_load=True):
        super().__init__(data_paths, batch_size, preprocessor, augmentations, include_ids, shuffle, prepare_on_load)
        self.noise_generator = noise_generator if noise_generator is not None else self.add_random_noise
        self.noise_probability = noise_probability

    @classmethod
    def noisy_mnist_train_val(cls, batch_size, target_shape, augmentations=None, shuffle=True, val_ratio=0.1,
                              split_seed=None, noise_prob=0.5, noise_generator=None):
        ims = mnist.train_images()
        if split_seed is not None:
            rs = np.random.RandomState(split_seed)
            ims_shuffled = [ims[i] for i in rs.permutation(len(ims))]
        else:
            ims_shuffled = [ims[i] for i in np.random.permutation(len(ims))]

        train_ims = ims_shuffled[int(val_ratio * len(ims_shuffled)):]
        val_ims = ims_shuffled[:int(val_ratio * len(ims_shuffled))]

        preprocessor = CNNPreProcessor(bgr_mean=0.1307, bgr_std=0.3081, target_shape=target_shape)
        train_ds = cls([], batch_size, preprocessor, augmentations=augmentations, shuffle=shuffle,
                       noise_generator=noise_generator, noise_probability=noise_prob)
        for idx, im in enumerate(train_ims):
            train_ds.data_d[f"{idx}"] = im
        val_ds = cls([], batch_size, preprocessor, shuffle=shuffle, noise_generator=noise_generator,
                     noise_probability=noise_prob)
        for idx, im in enumerate(val_ims):
            val_ds.data_d[f"{idx}"] = im
        return train_ds, val_ds

    def add_random_noise(self, im):
        if np.random.randn() > self.noise_probability:
            return im
        noise_type = np.random.choice([NoiseType.GAUSSIAN, NoiseType.SALT_AND_PEPPER, NoiseType.POISSON])
        return add_noise(noise_type, im)

    def batches(self):
        data = [v for k, v in sorted(self.data_d.items())]

        if self.shuffle:
            data = [data[i] for i in np.random.permutation(len(data))]

        curr_idx = 0
        while curr_idx < len(data):
            x_batch_list = []
            y_batch_list = []

            while curr_idx < len(data) and len(x_batch_list) < self.batch_size:
                x_im = data[curr_idx].copy()
                x_im = self.noise_generator(x_im)
                y_im = data[curr_idx].copy()

                if self.debug:
                    sbs = put_side_by_side([x_im, y_im])
                    cv.imshow("Before Augmentation", sbs)
                    cv.waitKey()
                    cv.destroyAllWindows()

                if self.augmentations is not None:
                    x_im, y_im = self.augmentations.augment(x_im, y_im)

                if self.debug and self.augmentations is not None:
                    sbs = put_side_by_side([x_im, y_im])
                    cv.imshow("After Augmentation", sbs)
                    cv.waitKey()
                    cv.destroyAllWindows()

                x_tensor = self.preprocessor.preprocess(x_im)
                x_batch_list.append(x_tensor)
                y_tensor = self.preprocessor.preprocess(y_im)
                y_batch_list.append(y_tensor)

                curr_idx += 1

            yield torch.cat(x_batch_list), torch.cat(y_batch_list)


def augmentation_experiments():
    aug = get_augmentations()

    test_ims_path = "agent_frames/cartpoleV0/"
    test_im_paths = [join(test_ims_path, fn) for fn in listdir(test_ims_path)]

    mnist_train_ds, mnist_val_ds = DeNoisingAutoEncoderDataset.noisy_mnist_train_val(1, (32, 32))
    mnist_train_ims = [el for el in mnist_train_ds.data_d.values()]
    mnist_train_ds.debug = True

    # for idx, batch in enumerate(mnist_train_ds.batches()):
    #     print(idx)

    for _ in range(100):
        # test_im_path = np.random.choice(test_im_paths)
        # print(test_im_path)
        #
        # test_im = cv.imread(test_im_path)
        test_im = mnist_train_ims[np.random.randint(len(mnist_train_ims))]

        test_im_x_out, test_im_y_out = aug.augment(test_im, test_im)

        sbs = put_side_by_side([test_im, test_im_x_out, test_im_y_out])
        opencv_show(sbs)


def seq_experiments():
    logging.basicConfig(level=logging.INFO)
    ims_path = "keyboard_agent_frames/MountainCar-v0"

    train_sequences = load_json("mountain_car_v0_train_sequences_31122020.json")
    val_sequences = load_json("mountain_car_v0_val_sequences_31122020.json")

    mountain_car_v0_64g_mean = 0.9857
    mountain_car_v0_64g_std = 0.1056

    target_shape = (64, 64)
    num_in_channels = 1
    in_size = (num_in_channels,) + target_shape

    batch_size = 32
    rnn_dim = 128
    num_actions = 3

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    preprocessor = CNNPreProcessor(bgr_mean=mountain_car_v0_64g_mean, bgr_std=mountain_car_v0_64g_std,
                                   target_shape=target_shape, to_grayscale=True)

    model = SequencePredictor(in_size, build_basic_encoder(num_in_channels), build_basic_decoder(num_in_channels),
                              rnn_dim, action_dim=num_actions - 1).to(device)

    dataset = SequencePredictionDataset([ims_path], batch_size, preprocessor, include_ids=train_sequences,
                                        num_actions=num_actions)

    for batch in dataset.batches():
        in_x, in_a, y = batch
        in_x = in_x.to(device)
        in_a = in_a.to(device)
        y = y.to(device)
        out = model(in_x, in_a)

        loss = F.mse_loss(y, out)
        loss.backward()
        print("")

    pass


if __name__ == "__main__":
    seq_experiments()

import logging
import sys
from datetime import datetime
from math import ceil, exp, log
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from data import DeNoisingAutoEncoderDataset, AutoEncoderDataset, get_augmentations, SequencePredictionDataset, \
    SequenceReconstructionDataset
from model import CNNPreProcessor, AutoEncoder, VariationalAutoEncoder, SequencePredictor, DeepVariationalBayesFilter
from utils import load_json, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class DummySummaryWriter(object):

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass


def kl_loss(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1))


def vae_loss(out, y, mu, log_var, kl_factor):
    return F.mse_loss(out, y) + kl_factor * kl_loss(mu, log_var)


class LRPickScheduler(_LRScheduler):

    def __init__(self, optimizer, lr_pick_steps=500, begin_lr=1e-5, end_lr=10.0, method="exponential",
                 kill_on_finish=False):
        self.optimizer = optimizer
        self.begin_lr = begin_lr
        self.set_lr(self.begin_lr)
        self.end_lr = end_lr
        self.lr_pick_steps = lr_pick_steps
        self.method = method
        self.kill_on_finish = kill_on_finish
        if method == "exponential":
            lr_range = ceil(self.end_lr / self.begin_lr)
            self.lr_step_size = exp(log(lr_range) / self.lr_pick_steps)
        elif method == "linear":
            self.lr_step_size = log(self.end_lr - self.begin_lr) / self.lr_pick_steps
        else:
            raise ValueError(f"Unknown method {method}")

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def set_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr

    def step(self, epoch=None):
        curr_lr = self.get_lr()
        if curr_lr >= self.end_lr:
            if self.kill_on_finish:
                logger.info("Picking LR finished. Exiting.")
                sys.exit()
            return

        if self.method == "exponential":
            self.set_lr(curr_lr * self.lr_step_size)
        elif self.method == "linear":
            self.set_lr(curr_lr + self.lr_step_size)
        else:
            raise ValueError(f"Unknown method {self.method}")


class ModelTrainer(object):

    def __init__(self, model, model_id, loss_function, optimizer, device, scheduler=None, checkpoint_path=None,
                 writer=None, batch_wise_scheduler=False):
        self.model = model
        self.model_id = model_id
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.writer = writer if writer is not None else DummySummaryWriter()
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0
        self.batch_wise_scheduler = batch_wise_scheduler

    @classmethod
    def get_lr_pick_trainer(cls, model, model_id, loss_function, optimizer, device, checkpoint_path=None,
                            writer=None, batch_wise_scheduler=True, lr_pick_steps=500, begin_lr=1e-5,
                            end_lr=10.0, method="exponential", kill_on_finish=True):
        trainer = cls(model, model_id, loss_function, optimizer, device,
                      scheduler=LRPickScheduler(optimizer, lr_pick_steps, begin_lr, end_lr, method, kill_on_finish),
                      checkpoint_path=checkpoint_path, writer=writer, batch_wise_scheduler=batch_wise_scheduler)
        trainer.validate = lambda x: None
        return trainer

    def scheduler_step(self):
        if self.scheduler is not None:
            current_lr = self.scheduler.optimizer.param_groups[0]["lr"]

            try:
                self.scheduler.step()
            except UnboundLocalError as e:  # For catching OneCycleLR errors when stepping too often
                return
            log_prefix = "batch" if self.batch_wise_scheduler else "epoch"
            log_idx = self.curr_train_batch_idx if self.batch_wise_scheduler else self.curr_epoch_idx
            self.writer.add_scalar(f"{log_prefix}/lr", current_lr, log_idx)

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            return

        filename = f"{self.model_id}_{self.curr_epoch_idx:03d}.tar"
        path = join(self.checkpoint_path, filename)
        save_checkpoint(path, self.model, self.optimizer, epoch=self.curr_epoch_idx)

    def calculate_loss(self, model_output, ground_truth):
        return self.loss_function(model_output, ground_truth)

    def get_inputs_and_ground_truth(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def fit(self, dataset_train, dataset_validate, num_epochs=10, training_seed=None):
        if training_seed is not None:
            np.random.seed(training_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(training_seed)

        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0

        logger.info(f"Starting Training For {num_epochs} epochs.")
        for _ in range(num_epochs):
            logger.info(f"Epoch {self.curr_epoch_idx}")
            logger.info(f"Training")
            self.train(dataset_train)
            logger.info(f"Validation")
            self.validate(dataset_validate)

            if not self.batch_wise_scheduler:
                self.scheduler_step()
            self.curr_epoch_idx += 1
            self.save_checkpoint()

    def train(self, dataset):
        self.model.train()

        epoch_loss = 0.0
        count_batches = 0

        for batch in dataset.batches():
            batch_loss = self.training_step(batch, self.curr_train_batch_idx)
            self.writer.add_scalar("train_batch/loss", batch_loss, self.curr_train_batch_idx)
            logger.info(
                f"Training - Epoch: {self.curr_epoch_idx} Batch: {self.curr_train_batch_idx}: Loss {batch_loss}")
            self.curr_train_batch_idx += 1
            count_batches += 1
            epoch_loss += batch_loss

        epoch_loss /= max(1.0, count_batches)
        self.writer.add_scalar("train_epoch/loss", epoch_loss, self.curr_epoch_idx)
        logger.info(f"Training - Epoch {self.curr_epoch_idx}: Loss {epoch_loss}")

    def model_output(self, model_in):
        return self.model(model_in)

    def training_step(self, batch, batch_idx):
        model_in, gt = self.get_inputs_and_ground_truth(batch)

        self.optimizer.zero_grad()
        model_out = self.model_output(model_in)
        loss = self.calculate_loss(model_out, gt)

        loss.backward()
        self.optimizer.step()
        if self.batch_wise_scheduler:
            self.scheduler_step()
        return loss.item()

    def validate(self, dataset):
        self.model.eval()

        epoch_loss = 0.0
        count_batches = 0

        with torch.no_grad():
            for batch in dataset.batches():
                batch_loss = self.validation_step(batch, self.curr_val_batch_idx)
                logger.info(
                    f"Validation - Epoch: {self.curr_epoch_idx} Batch: {self.curr_val_batch_idx}: Loss {batch_loss}")
                self.writer.add_scalar("val_batch/loss", batch_loss, self.curr_val_batch_idx)

                self.curr_val_batch_idx += 1
                count_batches += 1
                epoch_loss += batch_loss

        epoch_loss /= max(1.0, count_batches)

        self.writer.add_scalar("val_epoch/loss", epoch_loss, self.curr_epoch_idx)
        logger.info(f"Validation - Epoch {self.curr_epoch_idx}: Loss {epoch_loss}")

    def validation_step(self, batch, batch_idx):
        model_in, gt = self.get_inputs_and_ground_truth(batch)

        model_out = self.model_output(model_in)
        loss = self.calculate_loss(model_out, gt)

        return loss.item()


class AutoEncoderTrainer(ModelTrainer):

    def __init__(self, model, model_id, loss_function, optimizer, device, scheduler=None, checkpoint_path=None,
                 writer=None, batch_wise_scheduler=False):
        super().__init__(model, model_id, loss_function, optimizer, device, scheduler, checkpoint_path, writer,
                         batch_wise_scheduler)

    def get_inputs_and_ground_truth(self, batch):
        x = batch
        x = x.to(self.device)
        return x, x

    def calculate_loss(self, model_output, ground_truth):
        out, encoding = model_output
        return self.loss_function(out, ground_truth)


class DeNoisingAutoEncoderTrainer(ModelTrainer):

    def __init__(self, model, model_id, loss_function, optimizer, device, scheduler=None, checkpoint_path=None,
                 writer=None, batch_wise_scheduler=False):
        super().__init__(model, model_id, loss_function, optimizer, device, scheduler, checkpoint_path, writer,
                         batch_wise_scheduler)

    def calculate_loss(self, model_output, ground_truth):
        out, encoding = model_output
        return self.loss_function(out, ground_truth)


class VariationalAutoEncoderTrainer(ModelTrainer):

    def __init__(self, model, model_id, optimizer, device, scheduler=None, checkpoint_path=None,
                 writer=None, batch_wise_scheduler=False, kl_factor=0.005):
        super().__init__(model, model_id, vae_loss, optimizer, device, scheduler, checkpoint_path, writer,
                         batch_wise_scheduler)
        self.kl_factor = kl_factor

    @classmethod
    def get_lr_pick_trainer(cls, model, model_id, optimizer, device, checkpoint_path=None,
                            writer=None, batch_wise_scheduler=True, lr_pick_steps=500, begin_lr=1e-5,
                            end_lr=10.0, method="exponential", kill_on_finish=True, kl_factor=0.005):
        trainer = cls(model, model_id, optimizer, device,
                      scheduler=LRPickScheduler(optimizer, lr_pick_steps, begin_lr, end_lr, method, kill_on_finish),
                      checkpoint_path=checkpoint_path, writer=writer, batch_wise_scheduler=batch_wise_scheduler,
                      kl_factor=kl_factor)
        trainer.validate = lambda x: None
        return trainer

    def get_inputs_and_ground_truth(self, batch):
        x = batch
        x = x.to(self.device)
        return x, x

    def calculate_loss(self, model_output, ground_truth):
        out, mu, log_var = model_output
        loss = self.loss_function(out, ground_truth, mu, log_var, self.kl_factor)
        return loss


class DeepVariationalBayesFilterTrainer(ModelTrainer):

    def __init__(self, model, model_id, optimizer, device, scheduler=None, checkpoint_path=None,
                 writer=None, batch_wise_scheduler=False, with_actions=True, kl_factor=0.005):
        super().__init__(model, model_id, vae_loss, optimizer, device, scheduler, checkpoint_path, writer,
                         batch_wise_scheduler)
        self.with_actions = with_actions
        self.kl_factor = kl_factor

    @classmethod
    def get_lr_pick_trainer(cls, model, model_id, optimizer, device, checkpoint_path=None, writer=None,
                            batch_wise_scheduler=True, lr_pick_steps=500, begin_lr=1e-5,
                            end_lr=10.0, method="exponential", kill_on_finish=True, kl_factor=0.005):
        trainer = cls(model, model_id, optimizer, device,
                      scheduler=LRPickScheduler(optimizer, lr_pick_steps, begin_lr, end_lr, method, kill_on_finish),
                      checkpoint_path=checkpoint_path, writer=writer, batch_wise_scheduler=batch_wise_scheduler,
                      kl_factor=kl_factor)
        trainer.validate = lambda x: None
        return trainer

    def get_inputs_and_ground_truth(self, batch):
        x, a = batch
        x = x.to(self.device)
        a = a.to(self.device) if a is not None else None
        return (x, a), x

    def model_output(self, model_in):
        x, a = model_in
        return self.model(x, a)

    def calculate_loss(self, model_output, ground_truth):
        decoder_outs, z_s, w_s, w_mus, w_log_vars = model_output
        losses = [self.loss_function(decoder_outs[idx], ground_truth[idx], w_mus[idx], w_log_vars[idx], self.kl_factor)
                  for idx in range(len(decoder_outs))]
        return sum(losses) / len(losses)


class SequencePredictorTrainer(ModelTrainer):

    def __init__(self, model, model_id, loss_function, optimizer, device, scheduler=None, checkpoint_path=None,
                 writer=None, batch_wise_scheduler=False, with_actions=True):
        super().__init__(model, model_id, loss_function, optimizer, device, scheduler, checkpoint_path, writer,
                         batch_wise_scheduler)
        self.with_actions = with_actions

    def get_inputs_and_ground_truth(self, batch):
        x, a, y = batch
        x = x.to(self.device)
        a = a.to(self.device) if a is not None else None
        y = y.to(self.device)
        return (x, a), y

    def model_output(self, model_in):
        x, a = model_in
        return self.model(x, a)


def main():
    logging.basicConfig(level=logging.INFO)

    cart_pole_v0_bgr_mean = (0.9922, 0.9931, 0.9940)
    cart_pole_v0_bgr_std = (0.0791, 0.0741, 0.0703)

    mountain_car_v0_64g_mean = 0.9857
    mountain_car_v0_64g_std = 0.1056

    target_shape = (64, 64)

    batch_size = 32
    lr = 1e-6
    # max_lr = 0.001
    max_lr = 1e-4
    num_epochs = 50
    latent_dim = 128
    kl_factor = 0.005
    num_actions = 3
    bidirectional_rnn = True
    rnn_layers = 1

    seq_length = 60
    # pretrained_model_path = None
    # pretrained_model_path = "useful_models/squeeze_dvbf_seq_length_30_bidirectional_mountain_car_v0_max_lr_0.0005_11012021_073140_050.tar"
    # pretrained_model_path = "useful_models/squeeze_dvbf_pretrained_seq_length_100_bidirectional_mountain_car_v0_max_lr_0.0005_14012021_164801_009.tar"
    # pretrained_model_path = "useful_models/squeeze_dvbf_pretrained_seq_length_60_bidirectional_mountain_car_v0_max_lr_0.0005_14012021_142801_007.tar"
    # pretrained_model_path = "useful_models/squeeze_dvbf_pretrained_seq_length_60_bidirectional_mountain_car_v0_max_lr_0.00015_15012021_102204_046.tar"
    pretrained_model_path = "useful_models/squeeze_dvbf_pretrained_seq_length_60_bidirectional_mountain_car_v0_big_max_lr_0.00015_17012021_124000_010.tar"

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    preprocessor = CNNPreProcessor(bgr_mean=mountain_car_v0_64g_mean, bgr_std=mountain_car_v0_64g_std,
                                   target_shape=target_shape, to_grayscale=True)

    checkpoint_path = "model_checkpoints"

    dataset_id = "mountain_car_v0_big"
    data_files_path = "keyboard_agent_frames/MountainCar-v0"
    train_files = load_json("mountain_car_v0_big_train_sequences_17012021.json")
    val_files = load_json("mountain_car_v0_big_val_sequences_17012021.json")

    train_dataset = SequenceReconstructionDataset([data_files_path], batch_size, preprocessor, include_ids=train_files,
                                                  prepare_on_load=True, num_actions=num_actions, seq_length=seq_length)
    val_dataset = SequenceReconstructionDataset([data_files_path], batch_size, preprocessor, include_ids=val_files,
                                                prepare_on_load=True, num_actions=num_actions, seq_length=seq_length)

    augmentations = None
    input_shape = (1,) + target_shape

    encoder_filters = (32, 64, 128, 256, 512)

    action_dim = num_actions - 1
    pretrained = pretrained_model_path is not None

    model = DeepVariationalBayesFilter.get_squeeze_dvbf_ll(input_shape=input_shape, latent_dim=latent_dim,
                                                           action_dim=action_dim, rnn_layers=rnn_layers,
                                                           bidirectional_rnn=bidirectional_rnn).to(device)
    if pretrained:
        load_checkpoint(pretrained_model_path, model, device=device)
        logger.info(f"Loaded pretrained model from \"{pretrained_model_path}\"")

    model_id = f"squeeze_dvbf{'_pretrained' if pretrained else ''}_seq_length_{seq_length}{'_bidirectional' if bidirectional_rnn else ''}_{dataset_id}_max_lr_{max_lr}_{datetime.now():%d%m%Y_%H%M%S}"
    session_id = f"squeeze_dvbf_{target_shape[0]}x{target_shape[1]}_{dataset_id}_ldim_{latent_dim}_one_cycle_sched"

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr, total_steps=num_epochs * train_dataset.batches_per_epoch,
                           cycle_momentum=False)
    # scheduler = None
    batch_scheduler = True
    writer = SummaryWriter(comment=f"-{model_id}-{session_id}")
    trainer = DeepVariationalBayesFilterTrainer(model, model_id, optimizer, device, scheduler=scheduler,
                                                checkpoint_path=checkpoint_path, writer=writer,
                                                batch_wise_scheduler=batch_scheduler)
    # trainer = DeepVariationalBayesFilterTrainer.get_lr_pick_trainer(model, model_id, optimizer, device, writer=writer,
    #                                                                 batch_wise_scheduler=batch_scheduler)
    trainer.fit(train_dataset, val_dataset, num_epochs=num_epochs)

    print("")


if __name__ == "__main__":
    main()

from __future__ import division
import sys
import time

import torch
from tqdm import tqdm
tqdm.monitor_interval = 0

from utils import CheckpointDataLoader, CheckpointSaver
from collections import OrderedDict


class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    state_dict = checkpoint[model]
                    renamed_state_dict = OrderedDict()
                    # change the names in the state_dict to match the new layer
                    for key, value in state_dict.items():
                        if 'layer' in key:
                            names = key.split('.')
                            names[1:1] = ['hmr_layer']
                            new_key = '.'.join(n for n in names)
                            renamed_state_dict[new_key] = value
                        else:
                            renamed_state_dict[key] = value
                    self.models_dict[model].load_state_dict(renamed_state_dict, strict=False)

    @staticmethod
    def linear_rampup(current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length

    def train(self):
        """Training process."""
        ramp_step = 0
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):

            # ------------------ update image size intervals ----------------------
            self.train_ds.update_size_intervals(epoch)
            # ---------------------------------------------------------------------

            # ------------------ update batch size ----------------------
            if epoch == 0:
                batch_size = self.options.batch_size    # 24
            elif epoch == 1:
                batch_size = self.options.batch_size // 2   # 12
            else:
                batch_size = self.options.batch_size // 3   # 8
                if epoch == 3:
                    self.options.checkpoint_steps = 2000
            # ---------------------------------------------------------------------

            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)

            # init alphas
            if epoch <= 3:
                self.model.init_alphas(epoch+1, self.device)

            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):

                # ------------------ ramp consistency loss weight after updating the scale interval ----------------------
                if self.options.ramp == 'up':
                    total_ramp = (len(self.train_ds) // self.options.batch_size) * 5
                    self.consistency_loss_ramp = self.linear_rampup(ramp_step, total_ramp)
                    ramp_step += 1
                elif self.options.ramp == 'down':
                    total_ramp = (len(self.train_ds) // self.options.batch_size) * 5
                    consistency_loss_ramp = self.linear_rampup(ramp_step, total_ramp)
                    self.consistency_loss_ramp = 1.0 - consistency_loss_ramp
                    ramp_step += 1
                else:
                    self.consistency_loss_ramp = 1.0
                # ---------------------------------------------------------------------


                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) and k != 'sample_index' else v for k,v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0 and epoch >= 3:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count)
                        tqdm.write('Checkpoint saved')

                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

                # for the first 3 epochs, we only train half epoch
                if epoch == 0:
                    if (step + 1) == (len(self.train_ds) // (self.options.batch_size * 2)):
                        break
                elif epoch == 1:
                    if (step + 1) == (len(self.train_ds) // self.options.batch_size):
                        break
                elif epoch == 2:
                    if (step + 1) == (len(self.train_ds) // (self.options.batch_size * 2)) * 3:
                        break

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None

            # update learning rate if lr scheduler is epoch-based
            if self.lr_scheduler is not None and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ExponentialLR):
                if (epoch + 1) % 4 == 0:
                    self.lr_scheduler.step()
        return

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        pass

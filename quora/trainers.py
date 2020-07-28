import logging
from pathlib import Path
from pprint import pprint, pformat
from typing import Callable, Iterable

import torch
from torch.nn.utils import clip_grad_norm_

from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from allennlp.nn.util import move_to_device

from tensorboardX import SummaryWriter

from quora.lstm_lvm.model import SeparatedQuoraModel
from quora.model import QuoraModel


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self,
                 model: QuoraModel,
                 optimizer: torch.optim.Optimizer,
                 labeled_iterator: DataIterator,
                 unlabeled_iterator: DataIterator,
                 train_labeled_dataset: Iterable[Instance],
                 train_unlabeled_dataset: Iterable[Instance],
                 validation_dataset: Iterable[Instance],
                 summary_writer: SummaryWriter,
                 serialization_dir: str,
                 num_epochs: int,
                 iters_per_epoch: int,
                 write_summary_every: int,
                 validate_every: int,
                 warmup: int = 0,
                 patience: int = None,
                 clip_grad_max_norm: float = None,
                 kl_weight_scheduler: Callable[[int], float] = None,
                 cuda_device: int = -1,
                 early_stop: bool = True) -> None:
        self.model = model
        self.optimizer = optimizer
        self.labeled_iterator = labeled_iterator
        self.unlabeled_iterator = unlabeled_iterator
        self.train_labeled_dataset = train_labeled_dataset
        self.train_unlabeled_dataset = train_unlabeled_dataset
        self.validation_dataset = validation_dataset
        self.summary_writer = summary_writer
        self.write_summary_every = write_summary_every
        self.validate_every = validate_every
        self.serialization_dir = Path(serialization_dir)
        self.iters_per_epoch = iters_per_epoch
        self.num_epochs = num_epochs
        self.clip_grad_max_norm = clip_grad_max_norm
        self.patience = patience
        self.warmup = warmup
        if kl_weight_scheduler:
            self.kl_weight_scheduler = kl_weight_scheduler
        else:
            self.kl_weight_scheduler = lambda x: 1.0
        self.early_stop = early_stop

        self.cuda_device = cuda_device
        self.model.to(cuda_device)

        self.global_step = 0
        self.best_epoch = -1
        self.best_accuracy = 0

    def write_summary(self, prefix):
        metrics = self.model.get_metrics(reset=True)
        for k, v in metrics.items():
            if isinstance(v, dict):
                for v_k, v_v in v.items():
                    self.summary_writer.add_scalar(
                        tag=f'{prefix}/{k}/{v_k}', scalar_value=v_v,
                        global_step=self.global_step)
            else:
                self.summary_writer.add_scalar(
                    tag=f'{prefix}/{k}', scalar_value=v,
                    global_step=self.global_step)
        return metrics

    def maybe_write_summary(self, prefix):
        if self.global_step % self.write_summary_every == 0:
            self.write_summary(prefix=prefix)

    def train_epoch(self):
        self.model.get_metrics(reset=True)
        train_labeled_generator = self.labeled_iterator(
            instances=self.train_labeled_dataset, shuffle=True)
        if self.unlabeled_iterator is not None:
            train_unlabeled_generator = self.unlabeled_iterator(
                instances=self.train_unlabeled_dataset, shuffle=True)
        else:
            train_unlabeled_generator = None
        for _ in range(self.iters_per_epoch):
            self.model.train()
            labeled_batch = next(train_labeled_generator)
            labeled_batch = move_to_device(labeled_batch, self.cuda_device)
            labeled_output = self.model(
                premise=labeled_batch['premise'],
                hypothesis=labeled_batch['hypothesis'],
                label=labeled_batch['label'])
            loss = labeled_output['loss']

            if train_unlabeled_generator is not None:
                unlabeled_batch = next(train_unlabeled_generator)
                unlabeled_batch = move_to_device(unlabeled_batch, self.cuda_device)
                unlabeled_output = self.model(
                    premise=unlabeled_batch['premise'],
                    hypothesis=unlabeled_batch['hypothesis'],
                    label=None)
                loss = loss + unlabeled_output['loss']

            loss = loss + self.model.get_regularization_penalty()

            self.model.zero_grad()
            loss.backward()
            if self.clip_grad_max_norm is not None:
                clip_grad_norm_(self.model.parameters(),
                                max_norm=self.clip_grad_max_norm)
            self.optimizer.step()
            self.maybe_write_summary(prefix='train')

            if self.global_step % self.validate_every == 0:
                self.validate()

            self.global_step += 1
            new_kl_weight = self.kl_weight_scheduler(self.global_step)
            self.model.kl_weight = new_kl_weight

    def validate(self):
        self.model.eval()
        self.model.get_metrics(reset=True)
        valid_generator = self.labeled_iterator(
            instances=self.validation_dataset, shuffle=False, num_epochs=1)
        with torch.no_grad():
            for batch in valid_generator:
                batch = move_to_device(batch, self.cuda_device)
                self.model(premise=batch['premise'],
                           hypothesis=batch['hypothesis'],
                           label=batch['label'])
        metrics = self.write_summary(prefix='valid')
        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            model_filename = f'model_state_best.th'
        else:
            model_filename = f'model_state_epoch_{epoch}.th'
        model_path = self.serialization_dir / model_filename
        model_state = self.model.state_dict()
        torch.save(model_state, model_path)

    def train(self):
        orig_disc_weight = self.model.discriminator_weight
        for epoch in range(1, self.num_epochs + 1):
            if epoch >= self.warmup:
                self.model.discriminator_weight = orig_disc_weight
            else:
                self.model.discriminator_weight = 0
            self.train_epoch()
            valid_metrics = self.validate()
            logger.info(f'### Epoch {epoch} ###')
            logger.info(pformat(valid_metrics))
            # self.save_checkpoint(epoch)

            disc_acc = valid_metrics['labeled']['discriminator_accuracy']
            if disc_acc > self.best_accuracy:
                self.best_accuracy = disc_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch=epoch, is_best=True)
                logger.info('Saved the new best checkpoint')
            else:
                if (self.patience is not None
                        and self.best_epoch + self.patience < epoch):
                    if self.early_stop:
                        break


class FineTuningTrainer:

    def __init__(self,
                 model: QuoraModel,
                 main_optimizer: torch.optim.Optimizer,
                 aux_optimizer: torch.optim.Optimizer,
                 labeled_iterator: DataIterator,
                 unlabeled_iterator: DataIterator,
                 train_labeled_dataset: Iterable[Instance],
                 train_unlabeled_dataset: Iterable[Instance],
                 validation_dataset: Iterable[Instance],
                 summary_writer: SummaryWriter,
                 serialization_dir: str,
                 num_epochs: int,
                 iters_per_epoch: int,
                 write_summary_every: int,
                 validate_every: int,
                 patience: int = None,
                 clip_grad_max_norm: float = None,
                 kl_weight_scheduler: Callable[[int], float] = None,
                 gumbel_temperature_scheduler: Callable[[int], float] = None,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.main_optimizer = main_optimizer
        self.aux_optimizer = aux_optimizer
        self.labeled_iterator = labeled_iterator
        self.unlabeled_iterator = unlabeled_iterator
        self.train_labeled_dataset = train_labeled_dataset
        self.train_unlabeled_dataset = train_unlabeled_dataset
        self.validation_dataset = validation_dataset
        self.summary_writer = summary_writer
        self.write_summary_every = write_summary_every
        self.validate_every = validate_every
        self.serialization_dir = Path(serialization_dir)
        self.iters_per_epoch = iters_per_epoch
        self.num_epochs = num_epochs
        self.clip_grad_max_norm = clip_grad_max_norm
        self.patience = patience
        if kl_weight_scheduler:
            self.kl_weight_scheduler = kl_weight_scheduler
        else:
            self.kl_weight_scheduler = lambda x: 1.0
        if gumbel_temperature_scheduler:
            self.gumbel_temperature_scheduler = gumbel_temperature_scheduler
        else:
            self.gumbel_temperature_scheduler = lambda x: 1.0

        self.cuda_device = cuda_device
        self.model.to(cuda_device)

        self.global_step = 0
        self.best_epoch = -1
        self.best_accuracy = 0

    def write_summary(self, prefix):
        metrics = self.model.get_metrics(reset=True)
        for k, v in metrics.items():
            if isinstance(v, dict):
                for v_k, v_v in v.items():
                    self.summary_writer.add_scalar(
                        tag=f'{prefix}/{k}/{v_k}', scalar_value=v_v,
                        global_step=self.global_step)
            else:
                self.summary_writer.add_scalar(
                    tag=f'{prefix}/{k}', scalar_value=v,
                    global_step=self.global_step)
        return metrics

    def maybe_write_summary(self, prefix):
        if self.global_step % self.write_summary_every == 0:
            self.write_summary(prefix=prefix)

    def train_epoch(self):
        self.model.get_metrics(reset=True)
        train_labeled_generator = self.labeled_iterator(
            instances=self.train_labeled_dataset, shuffle=True)
        train_unlabeled_generator = self.unlabeled_iterator(
            instances=self.train_unlabeled_dataset, shuffle=True)
        for _ in range(self.iters_per_epoch):
            self.model.train()
            labeled_batch = next(train_labeled_generator)
            unlabeled_batch = next(train_unlabeled_generator)
            labeled_batch = move_to_device(labeled_batch, self.cuda_device)
            unlabeled_batch = move_to_device(unlabeled_batch, self.cuda_device)
            labeled_output = self.model(
                premise=labeled_batch['premise'],
                hypothesis=labeled_batch['hypothesis'],
                label=labeled_batch['label'])
            unlabeled_output = self.model(
                premise=unlabeled_batch['premise'],
                hypothesis=unlabeled_batch['hypothesis'],
                label=None)
            loss = labeled_output['loss'] + unlabeled_output['loss']
            loss = loss + self.model.get_regularization_penalty()

            self.model.zero_grad()
            loss.backward()
            if self.clip_grad_max_norm is not None:
                clip_grad_norm_(self.model.parameters(),
                                max_norm=self.clip_grad_max_norm)
            self.main_optimizer.step()

            aux_labeled_output = self.model.aux_forward(
                premise=labeled_batch['premise'],
                hypothesis=labeled_batch['hypothesis'])
            aux_loss = aux_labeled_output['loss']
            if train_unlabeled_generator is not None:
                aux_unlabeled_output = self.model.aux_forward(
                    premise=unlabeled_batch['premise'],
                    hypothesis=unlabeled_batch['hypothesis'])
                aux_loss = aux_loss + aux_unlabeled_output['loss']
            self.model.zero_grad()
            aux_loss.backward()
            if self.clip_grad_max_norm is not None:
                clip_grad_norm_(self.model.parameters(),
                                max_norm=self.clip_grad_max_norm)
            self.aux_optimizer.step()

            self.maybe_write_summary(prefix='train')

            if self.global_step % self.validate_every == 0:
                self.validate()

            self.global_step += 1

            new_kl_weight = self.kl_weight_scheduler(self.global_step)
            self.model.kl_weight = new_kl_weight

            new_gumbel_temperature = self.gumbel_temperature_scheduler(
                self.global_step)
            self.model.gumbel_temperature = new_gumbel_temperature

    def validate(self):
        self.model.eval()
        self.model.get_metrics(reset=True)
        valid_generator = self.labeled_iterator(
            instances=self.validation_dataset, shuffle=False, num_epochs=1)
        with torch.no_grad():
            for batch in valid_generator:
                batch = move_to_device(batch, self.cuda_device)
                self.model(premise=batch['premise'],
                           hypothesis=batch['hypothesis'],
                           label=batch['label'])
        metrics = self.write_summary(prefix='valid')
        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            model_filename = f'model_state_best.th'
        else:
            model_filename = f'model_state_epoch_{epoch}.th'
        model_path = self.serialization_dir / model_filename
        model_state = self.model.state_dict()
        torch.save(model_state, model_path)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch()
            valid_metrics = self.validate()
            logger.info(f'### Epoch {epoch} ###')
            logger.info(pformat(valid_metrics))
            # self.save_checkpoint(epoch)

            disc_acc = valid_metrics['labeled']['discriminator_accuracy']
            if disc_acc > self.best_accuracy:
                self.best_accuracy = disc_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch=epoch, is_best=True)
                logger.info('Saved the new best checkpoint')
            else:
                if (self.patience is not None
                        and self.best_epoch + self.patience < epoch):
                    break
            if self.model.con_autoweight:
                logger.info(f'con_y_weight_p: {self.model.con_y_weight_p.item()}')
                logger.info(f'con_z_weight_p: {self.model.con_z_weight_p.item()}')
                logger.info(f'con_z2_weight_p: {self.model.con_z2_weight_p.item()}')


class SupervisedTrainer:

    def __init__(self,
                 model: QuoraModel,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Iterable[Instance],
                 summary_writer: SummaryWriter,
                 serialization_dir: str,
                 num_epochs: int,
                 iters_per_epoch: int,
                 write_summary_every: int,
                 validate_every: int,
                 patience: int = None,
                 clip_grad_max_norm: float = None,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.optimizer = optimizer
        self.iterator = iterator
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.summary_writer = summary_writer
        self.write_summary_every = write_summary_every
        self.validate_every = validate_every
        self.serialization_dir = Path(serialization_dir)
        self.iters_per_epoch = iters_per_epoch
        self.num_epochs = num_epochs
        self.clip_grad_max_norm = clip_grad_max_norm
        self.patience = patience

        self.cuda_device = cuda_device
        self.model.to(cuda_device)

        self.global_step = 0
        self.best_epoch = -1
        self.best_accuracy = 0

    def write_summary(self, prefix):
        metrics = self.model.get_metrics(reset=True)
        for k, v in metrics.items():
            if isinstance(v, dict):
                for v_k, v_v in v.items():
                    self.summary_writer.add_scalar(
                        tag=f'{prefix}/{k}/{v_k}', scalar_value=v_v,
                        global_step=self.global_step)
            else:
                self.summary_writer.add_scalar(
                    tag=f'{prefix}/{k}', scalar_value=v,
                    global_step=self.global_step)
        return metrics

    def maybe_write_summary(self, prefix):
        if self.global_step % self.write_summary_every == 0:
            self.write_summary(prefix=prefix)

    def train_epoch(self):
        self.model.get_metrics(reset=True)
        train_generator = self.iterator(
            instances=self.train_dataset, shuffle=True)
        for _ in range(self.iters_per_epoch):
            self.model.train()
            batch = next(train_generator)
            batch = move_to_device(batch, self.cuda_device)
            output = self.model(
                premise=batch['premise'],
                hypothesis=batch['hypothesis'],
                label=batch['label'])
            loss = output['loss']

            self.model.zero_grad()
            loss.backward()
            if self.clip_grad_max_norm is not None:
                clip_grad_norm_(self.model.parameters(),
                                max_norm=self.clip_grad_max_norm)
            self.optimizer.step()
            self.maybe_write_summary(prefix='train')

            if self.global_step % self.validate_every == 0:
                self.validate()

            self.global_step += 1

    def validate(self):
        self.model.eval()
        self.model.get_metrics(reset=True)
        valid_generator = self.iterator(
            instances=self.validation_dataset, shuffle=False, num_epochs=1)
        with torch.no_grad():
            for batch in valid_generator:
                batch = move_to_device(batch, self.cuda_device)
                self.model(premise=batch['premise'],
                           hypothesis=batch['hypothesis'],
                           label=batch['label'])
        metrics = self.write_summary(prefix='valid')
        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            model_filename = f'model_state_best.th'
        else:
            model_filename = f'model_state_epoch_{epoch}.th'
        model_path = self.serialization_dir / model_filename
        model_state = self.model.state_dict()
        torch.save(model_state, model_path)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch()
            valid_metrics = self.validate()
            print(f'### Epoch {epoch} ###')
            pprint(valid_metrics)
            self.save_checkpoint(epoch)

            disc_acc = valid_metrics['labeled']['discriminator_accuracy']
            if disc_acc > self.best_accuracy:
                self.best_accuracy = disc_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch=epoch, is_best=True)
                logger.info('Saved the new best checkpoint')
            else:
                if (self.patience is not None
                        and self.best_epoch + self.patience < epoch):
                    break


class SeparatedLVMTrainer:

    def __init__(self,
                 model: SeparatedQuoraModel,
                 optimizer: torch.optim.Optimizer,
                 labeled_iterator: DataIterator,
                 unlabeled_iterator: DataIterator,
                 train_labeled_dataset: Iterable[Instance],
                 train_unlabeled_dataset: Iterable[Instance],
                 validation_dataset: Iterable[Instance],
                 summary_writer: SummaryWriter,
                 serialization_dir: str,
                 num_epochs: int,
                 iters_per_epoch: int,
                 write_summary_every: int,
                 validate_every: int,
                 patience: int = None,
                 clip_grad_max_norm: float = None,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.optimizer = optimizer
        self.labeled_iterator = labeled_iterator
        self.unlabeled_iterator = unlabeled_iterator
        self.train_labeled_dataset = train_labeled_dataset
        self.train_unlabeled_dataset = train_unlabeled_dataset
        self.validation_dataset = validation_dataset
        self.summary_writer = summary_writer
        self.write_summary_every = write_summary_every
        self.validate_every = validate_every
        self.serialization_dir = Path(serialization_dir)
        self.iters_per_epoch = iters_per_epoch
        self.num_epochs = num_epochs
        self.clip_grad_max_norm = clip_grad_max_norm
        self.patience = patience

        self.cuda_device = cuda_device
        self.model.to(cuda_device)

        self.global_step = 0
        self.best_epoch = -1
        self.best_accuracy = 0

    def write_summary(self, prefix):
        metrics = self.model.get_metrics(reset=True)
        for k, v in metrics.items():
            if isinstance(v, dict):
                for v_k, v_v in v.items():
                    self.summary_writer.add_scalar(
                        tag=f'{prefix}/{k}/{v_k}', scalar_value=v_v,
                        global_step=self.global_step)
            else:
                self.summary_writer.add_scalar(
                    tag=f'{prefix}/{k}', scalar_value=v,
                    global_step=self.global_step)
        return metrics

    def maybe_write_summary(self, prefix):
        if self.global_step % self.write_summary_every == 0:
            self.write_summary(prefix=prefix)

    def train_epoch(self):
        self.model.get_metrics(reset=True)
        train_labeled_generator = self.labeled_iterator(
            instances=self.train_labeled_dataset, shuffle=True)
        if self.unlabeled_iterator is not None:
            train_unlabeled_generator = self.unlabeled_iterator(
                instances=self.train_unlabeled_dataset, shuffle=True)
        else:
            train_unlabeled_generator = None

        for _ in range(self.iters_per_epoch):
            self.model.train()
            labeled_batch = next(train_labeled_generator)
            labeled_batch = move_to_device(labeled_batch, self.cuda_device)
            labeled_output = self.model(
                premise=labeled_batch['premise'],
                hypothesis=labeled_batch['hypothesis'],
                label=labeled_batch['label'])
            labeled_loss = labeled_output['loss']
            loss = labeled_loss

            if self.unlabeled_iterator is not None:
                unlabeled_batch = next(train_unlabeled_generator)
                unlabeled_batch = move_to_device(
                    unlabeled_batch, self.cuda_device)
                unlabeled_output = self.model(
                    premise=unlabeled_batch['premise'],
                    hypothesis=unlabeled_batch['hypothesis'],
                    label=None)
                unlabeled_loss = unlabeled_output['loss']
                loss = loss + unlabeled_loss

            self.model.zero_grad()
            loss.backward()
            if self.clip_grad_max_norm is not None:
                clip_grad_norm_(self.model.parameters(),
                                max_norm=self.clip_grad_max_norm)
            self.optimizer.step()
            self.maybe_write_summary(prefix='train')

            if self.global_step % self.validate_every == 0:
                self.validate()

            self.global_step += 1

    def validate(self):
        self.model.eval()
        self.model.get_metrics(reset=True)
        valid_generator = self.labeled_iterator(
            instances=self.validation_dataset, shuffle=False, num_epochs=1)
        with torch.no_grad():
            for batch in valid_generator:
                batch = move_to_device(batch, self.cuda_device)
                self.model(premise=batch['premise'],
                           hypothesis=batch['hypothesis'],
                           label=batch['label'])
        metrics = self.write_summary(prefix='valid')
        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            model_filename = f'model_state_best.th'
        else:
            model_filename = f'model_state_epoch_{epoch}.th'
        model_path = self.serialization_dir / model_filename
        model_state = self.model.state_dict()
        torch.save(model_state, model_path)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch()
            valid_metrics = self.validate()
            print(f'### Epoch {epoch} ###')
            pprint(valid_metrics)

            disc_acc = valid_metrics['discriminator_accuracy']
            if disc_acc > self.best_accuracy:
                self.best_accuracy = disc_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch=epoch, is_best=True)
                logger.info('Saved the new best checkpoint')
            else:
                if (self.patience is not None
                        and self.best_epoch + self.patience < epoch):
                    break

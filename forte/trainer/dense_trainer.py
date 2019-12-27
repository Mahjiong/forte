# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Iterator
import functools

import torch
import torch.nn.functional as F

import texar.torch as tx
from texar.torch import HParams
from forte.common import Resources
from forte.data import MultiPack
from forte.models import BertBasedReranker
from forte.trainer.base.base_trainer import BaseTrainer


class DenseTrainer(BaseTrainer):
    def __init__(self):  # pylint: disable=unused-argument
        super().__init__()
        self._stop_train = False
        self._validation_requested = False
        self.train_instances = []

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resource: Resources, configs: HParams):
        """
        The training pipeline will run this initialization method during
        the initialization phase and send resources in as parameters.
        Args:

        Returns:

        """
        self.resource = resource
        self.config = configs
        # todo: Do it in a config specific way
        self.model = BertBasedReranker()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.model.to(device=self.device)

        # Builds learning rate decay scheduler
        static_lr = 1e-6
        num_train_steps = 400000
        num_warmup_steps = 40000

        vars_with_decay = []
        vars_without_decay = []
        for name, param in self.model.named_parameters():
            if 'layer_norm' in name or name.endswith('bias'):
                vars_without_decay.append(param)
            else:
                vars_with_decay.append(param)

        opt_params = [{
            'params': vars_with_decay,
            'weight_decay': 0.01,
        }, {
            'params': vars_without_decay,
            'weight_decay': 0.0,
        }]
        self.optim = tx.core.BertAdam(
            opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, functools.partial(self.get_lr_multiplier,
                                          total_steps=num_train_steps,
                                          warmup_steps=num_warmup_steps))

    # pylint: disable=no-self-use
    def get_lr_multiplier(self, step: int, total_steps: int,
                          warmup_steps: int) -> float:
        r"""Calculate the learning rate multiplier given current step and the
        number of warm-up steps. The learning rate schedule follows a linear
        warm-up and linear decay.
        """

        step = min(step, total_steps)

        multiplier = (1 - (step - warmup_steps) / (total_steps - warmup_steps))

        if warmup_steps > 0 and step < warmup_steps:
            warmup_percent_done = step / warmup_steps
            multiplier = warmup_percent_done

        return multiplier

    def data_request(self):
        pass

    def get_loss(self, instances: Iterator[Dict]):
        pass

    def train(self):
        self.model.train()

    def consume(self, m_pack: MultiPack):
        # consume the instance
        query = m_pack.get_pack(self.config.query_pack)
        pos_pack = m_pack.get_pack(self.config.positive_pack)
        neg_pack = m_pack.get_pack(self.config.negative_pack)
        pos_example = self.model.encode_text(text_a=query.text,
                                             text_b=pos_pack.text) + (1,)
        self.train_instances.append(pos_example)
        neg_example = self.model.encode_text(text_a=query.text,
                                             text_b=neg_pack.text) + (0,)
        self.train_instances.append(neg_example)

        if len(self.train_instances) % self.config.batch_size == 0:
            self.step()
            self.train_instances = []

    def step(self):
        self.optim.zero_grad()
        input_ids, segment_ids, input_mask, labels = zip(*self.train_instances)
        input_ids = torch.tensor(input_ids, device=self.device)
        segment_ids = torch.tensor(segment_ids, device=self.device)
        input_mask = torch.tensor(input_mask, device=self.device)
        input_lengths = (input_mask == 1).sum(dim=-1)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        logits, _ = self.model(input_ids=input_ids,
                               sequence_len=input_lengths,
                               segment_ids=segment_ids)
        loss = self._compute_loss(logits, labels)
        loss.backward()
        self.optim.step()
        self.scheduler.step()
        if self.scheduler.last_epoch % 500 == 0:
            print(f"Loss = {loss}, step = {self.scheduler.last_epoch}")

    def post_validation_action(self, dev_res):
        """
        This method
        Returns:

        """
        pass

    def _compute_loss(self, logits, labels):
        loss = F.cross_entropy(logits.view(-1, self.config.num_classes),
                               labels.view(-1), reduction='mean')
        return loss

    def update_resource(self):
        pass

    def pack_finish_action(self, pack_count: int):
        pass

    def epoch_finish_action(self, epoch_num: int):
        pass

    def request_eval(self):
        r"""The trainer should call this method to inform the pipeline to
        conduct evaluation.

        Returns:

        """
        self._validation_requested = True

    def request_stop_train(self):
        """
        The trainer should call this method to inform the pipeline to stop
        training.
        Returns:

        """
        self._stop_train = True

    def validation_done(self):
        r"""Used only by the pipeline to close the validation request.

        Returns:

        """
        self._validation_requested = False

    def validation_requested(self) -> bool:
        r"""Used only by the pipeline to check whether the trainer has made
        the validation request.

        Returns: True if the validation request is submitted and not completed.
        """
        return self._validation_requested

    def stop_train(self) -> bool:
        r"""Used only by the pipeline to check if the trainer decided to stop
        training.

        Returns: True if the trainer decided to stop.
        """
        # return self._stop_train
        return True

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
from typing import Any, Dict

import texar.torch as tx


class BertBasedReranker(tx.ModuleBase):

    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super().__init__()

        self.model = tx.modules.BERTClassifier(
            pretrained_model_name=pretrained_model_name)

        self.tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name=pretrained_model_name)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return tx.modules.BERTEncoder.default_hparams()

    def encode_text(self, text_a: str, text_b: str):
        max_query_length = 64
        max_seq_length = 512

        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token

        a_tokens = self.tokenizer.map_text_to_id(text_a)[:max_query_length - 2]
        a_tokens = [self.tokenizer.map_token_to_id(cls_token)] + \
                    a_tokens + [self.tokenizer.map_token_to_id(sep_token)]
        b_tokens = self.tokenizer.map_text_to_id(text_b)[:max_seq_length -
                                                          len(a_tokens) - 1]
        b_tokens = b_tokens + [self.tokenizer.map_token_to_id(sep_token)]
        input_length = len(a_tokens) + len(b_tokens)
        input_ids = a_tokens + b_tokens + [0] * (max_seq_length - input_length)
        segment_ids = [0] * len(a_tokens) + [1] * len(b_tokens) + \
                      [0] * (max_seq_length - input_length)
        input_mask = [1] * input_length + [0] * (max_seq_length - input_length)

        assert len(input_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, segment_ids, input_mask

    @property
    def output_size(self):
        return 0

    def forward(self, input_ids, sequence_len, segment_ids):
        logits, preds = self.model(inputs=input_ids,
                                   sequence_length=sequence_len,
                                   segment_ids=segment_ids)

        return logits, preds

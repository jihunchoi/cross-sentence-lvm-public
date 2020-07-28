from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.nn.activations import Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

from utils.metrics import ScalarMetric


class BaselineModel(Model):

    _NUM_LABELS = 2

    def __init__(self,
                 params: Params,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab=vocab)

        enc_hidden_dim = params.pop_int('enc_hidden_dim', 300)
        disc_hidden_dim = params.pop_int('disc_hidden_dim', 1200)
        disc_num_layers = params.pop_int('disc_num_layers', 1)

        emb_dropout = params.pop_float('emb_dropout', 0.0)
        disc_dropout = params.pop_float('disc_dropout', 0.0)
        l2_weight = params.pop_float('l2_weight', 0.0)

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.disc_dropout = nn.Dropout(disc_dropout)
        self._l2_weight = l2_weight

        self._token_embedder = Embedding.from_params(
            vocab=vocab, params=params.pop('token_embedder'))
        self._discriminator_encoder = PytorchSeq2VecWrapper(
            nn.LSTM(input_size=self._token_embedder.get_output_dim(),
                    hidden_size=enc_hidden_dim, batch_first=True))
        self._discriminator = FeedForward(
            input_dim=2 * self._discriminator_encoder.get_output_dim(),
            hidden_dims=[disc_hidden_dim]*disc_num_layers + [self._NUM_LABELS],
            num_layers=disc_num_layers + 1,
            activations=[Activation.by_name('relu')()] * disc_num_layers
                        + [Activation.by_name('linear')()])

        # Metrics
        self._metrics = {
            'labeled': {
                'discriminator_entropy': ScalarMetric(),
                'discriminator_accuracy': CategoricalAccuracy(),
                'loss': ScalarMetric()
            }
        }

    def get_regularization_penalty(self):
        sum_sq = sum(p.pow(2).sum() for p in self.parameters())
        l2_norm = sum_sq.sqrt()
        return self.l2_weight * l2_norm

    @property
    def l2_weight(self):
        return self._l2_weight

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._token_embedder(tokens)

    def discriminator_encode(self,
               inputs: torch.Tensor,
               mask: torch.Tensor,
               drop_start_token: bool = True) -> torch.Tensor:
        if drop_start_token:
            inputs = inputs[:, 1:]
            mask = mask[:, 1:]
        enc_hidden = self._discriminator_encoder(inputs.contiguous(), mask)
        return enc_hidden


    def discriminate(self,
                     premise_hidden: torch.Tensor,
                     hypothesis_hidden: torch.Tensor) -> torch.Tensor:
        disc_input = torch.cat(
            [premise_hidden * hypothesis_hidden,
             (premise_hidden - hypothesis_hidden).abs()],
            dim=-1)
        disc_input = self.disc_dropout(disc_input)
        disc_logits = self._discriminator(disc_input)
        return disc_logits

    def forward(self,
                premise: Dict[str, torch.Tensor],
                hypothesis: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        premise and hypothesis are padded with
        the BOS and the EOS token.
        """
        pre_mask = get_text_field_mask(premise)
        hyp_mask = get_text_field_mask(hypothesis)
        pre_tokens = premise['tokens']
        hyp_tokens = hypothesis['tokens']
        pre_token_embs = self.embed(pre_tokens)
        hyp_token_embs = self.embed(hyp_tokens)
        pre_token_embs = self.emb_dropout(pre_token_embs)
        hyp_token_embs = self.emb_dropout(hyp_token_embs)

        output_dict = {}

        pre_disc_hidden = self.discriminator_encode(
            inputs=pre_token_embs, mask=pre_mask, drop_start_token=True)
        hyp_disc_hidden = self.discriminator_encode(
            inputs=hyp_token_embs, mask=hyp_mask, drop_start_token=True)
        disc_logits = self.discriminate(
            premise_hidden=pre_disc_hidden,
            hypothesis_hidden=hyp_disc_hidden)
        disc_dist = Categorical(logits=disc_logits)
        disc_entropy = disc_dist.entropy().mean()
        disc_loss = functional.cross_entropy(
            input=disc_logits, target=label)

        loss = disc_loss

        output_dict['discriminator_entropy'] = disc_entropy
        output_dict['loss'] = loss

        self._metrics['labeled']['discriminator_entropy'](disc_entropy)
        self._metrics['labeled']['discriminator_accuracy'](
            predictions=disc_logits, gold_labels=label)
        self._metrics['labeled']['loss'](loss)

        return output_dict

    def get_metrics(self, reset: bool = False
                    ) -> Dict[str, Union[float, Dict[str, float]]]:
        metrics = {label_type: {k: v.get_metric(reset=reset)
                                for k, v in label_metrics.items()}
                   for label_type, label_metrics in self._metrics.items()}
        return metrics


def test():
    from pprint import pprint

    params = Params({
        'token_embedder': {
            'num_embeddings': 4,
            'embedding_dim': 3
        }
    })
    vocab = Vocabulary()
    while True:
        vocab_size = vocab.get_vocab_size()
        if vocab_size == 4:
            break
        vocab.add_token_to_namespace('a' + str(vocab_size))
    model = BaselineModel(params=params, vocab=vocab)
    premise = {'tokens': torch.randint(low=0, high=4, size=(5, 6))}
    hypothesis = {'tokens': torch.randint(low=0, high=4, size=(5, 7))}
    label = torch.randint(low=0, high=3, size=(5,))
    output = model(premise=premise, hypothesis=hypothesis, label=label)
    pprint(output)
    pprint(model.get_metrics())


if __name__ == '__main__':
    test()

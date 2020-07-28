from typing import Any, Dict, Optional, Union, List

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.nn.activations import Activation
from allennlp.nn.util import (
    get_text_field_mask, sequence_cross_entropy_with_logits
)
from allennlp.training.metrics import CategoricalAccuracy

from modules.code_generators import GaussianCodeGenerator, VmfCodeGenerator
from utils.metrics import ScalarMetric


class SeparatedQuoraModel(Model):

    _NUM_LABELS = 2

    def __init__(self,
                 params: Params,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab=vocab)

        enc_hidden_dim = params.pop_int('enc_hidden_dim', 300)
        gen_hidden_dim = params.pop_int('gen_hidden_dim', 300)
        disc_hidden_dim = params.pop_int('disc_hidden_dim', 1200)
        disc_num_layers = params.pop_int('disc_num_layers', 1)
        code_dist_type = params.pop_choice(
            'code_dist_type', ['gaussian', 'vmf'],
            default_to_first_choice=True)
        code_dim = params.pop_int('code_dim', 300)
        tie_embedding = params.pop_bool('tie_embedding', False)

        emb_dropout = params.pop_float('emb_dropout', 0.0)
        disc_dropout = params.pop_float('disc_dropout', 0.0)
        l2_weight = params.pop_float('l2_weight', 0.0)

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.disc_dropout = nn.Dropout(disc_dropout)
        self._l2_weight = l2_weight

        self._token_embedder = Embedding.from_params(
            vocab=vocab, params=params.pop('token_embedder'))
        self._encoder = PytorchSeq2VecWrapper(
            nn.LSTM(input_size=self._token_embedder.get_output_dim(),
                    hidden_size=enc_hidden_dim, batch_first=True))
        self._generator = PytorchSeq2SeqWrapper(
            nn.LSTM(input_size=(self._token_embedder.get_output_dim()
                                + code_dim),
                    hidden_size=gen_hidden_dim, batch_first=True))
        self._generator_projector = nn.Linear(
            in_features=self._generator.get_output_dim(),
            out_features=vocab.get_vocab_size())
        if tie_embedding:
            self._generator_projector.weight = self._token_embedder.weight

        if code_dist_type == 'vmf':
            vmf_kappa = params.pop_int('vmf_kappa', 150)
            self._code_generator = VmfCodeGenerator(
                input_dim=self._encoder.get_output_dim(),
                code_dim=code_dim, kappa=vmf_kappa)
        elif code_dist_type == 'gaussian':
            self._code_generator = GaussianCodeGenerator(
                input_dim=self._encoder.get_output_dim(),
                code_dim=code_dim)
        else:
            raise ValueError('Unknown code_dist_type')

        self._discriminator = FeedForward(
            input_dim=2 * self._code_generator.get_output_dim(),
            hidden_dims=[disc_hidden_dim]*disc_num_layers + [self._NUM_LABELS],
            num_layers=disc_num_layers + 1,
            activations=[Activation.by_name('relu')()] * disc_num_layers
                        + [Activation.by_name('linear')()],
            dropout=disc_dropout)

        self._kl_weight = 1.0
        self._discriminator_weight = params.pop_float(
            'discriminator_weight', 0.1)
        self._gumbel_temperature = 1.0

        # Metrics
        self._metrics = {
            'generator_loss': ScalarMetric(),
            'kl_divergence': ScalarMetric(),
            'discriminator_accuracy': CategoricalAccuracy(),
            'discriminator_loss': ScalarMetric(),
            'loss': ScalarMetric()
        }

    def get_regularization_penalty(self):
        sum_sq = sum(p.pow(2).sum() for p in self.parameters())
        l2_norm = sum_sq.sqrt()
        return self.l2_weight * l2_norm

    @property
    def l2_weight(self):
        return self._l2_weight

    @property
    def kl_weight(self):
        return self._kl_weight

    @kl_weight.setter
    def kl_weight(self, value):
        self._kl_weight = value

    @property
    def discriminator_weight(self):
        return self._discriminator_weight

    @discriminator_weight.setter
    def discriminator_weight(self, value):
        self._discriminator_weight = value

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._token_embedder(tokens)

    def encode(self,
               inputs: torch.Tensor,
               mask: torch.Tensor,
               drop_start_token: bool = True) -> torch.Tensor:
        if drop_start_token:
            inputs = inputs[:, 1:]
            mask = mask[:, 1:]
        enc_hidden = self._encoder(inputs.contiguous(), mask)
        return enc_hidden

    def sample_code_and_compute_kld(self,
                                    hidden: torch.Tensor) -> torch.Tensor:
        return self._code_generator(hidden)

    def discriminate(self,
                     premise_hidden: torch.Tensor,
                     hypothesis_hidden: torch.Tensor) -> torch.Tensor:
        disc_input = torch.cat(
            [premise_hidden + hypothesis_hidden,
             (premise_hidden - hypothesis_hidden).abs()],
            dim=-1)
        disc_input = self.disc_dropout(disc_input)
        disc_logits = self._discriminator(disc_input)
        return disc_logits

    def construct_generator_inputs(self,
                                   embeddings: torch.Tensor,
                                   code: torch.Tensor) -> torch.Tensor:
        batch_size, max_length, _ = embeddings.shape
        code_expand = code.unsqueeze(1).expand(
            batch_size, max_length, -1)
        inputs = torch.cat([embeddings, code_expand], dim=-1)
        return inputs

    def generate(self,
                 code: torch.Tensor,
                 max_length: torch.Tensor) -> torch.Tensor:
        start_index = self.vocab.get_token_index('<s>')
        end_index = self.vocab.get_token_index('</s>')
        pad_index = 0
        done = torch.zeros_like(max_length).long()
        max_max_length = max_length.max().item()

        prev_word = (torch.empty_like(done).long().unsqueeze(1)
                     .fill_(start_index))
        generated = []
        self._generator.stateful = True
        self._generator.reset_states()
        for t in range(max_max_length):
            if done.byte().all():
                break
            prev_word_emb = self.embed(prev_word)
            input_t = self.construct_generator_inputs(
                embeddings=prev_word_emb, code=code)
            hidden_t = self._generator(input_t, 1 - done.unsqueeze(1))
            pred_t = self._generator_projector(hidden_t).argmax(2)
            pred_t.masked_fill_(done.byte(), pad_index)
            generated.append(pred_t)
            done.masked_fill_(pred_t.eq(end_index).squeeze(1), 1)
            done.masked_fill_(max_length.le(t + 1), 1)
            prev_word = pred_t
        self._generator.stateful = False
        generated = torch.cat(generated, dim=1)
        return generated

    def convert_to_readable_text(self,
                                 generated: torch.Tensor) -> List[List[str]]:
        sequences = [seq.cpu().tolist() for seq in generated.unbind(0)]
        readable_sequences = []
        for seq in sequences:
            readable_seq = []
            for word_index in seq:
                if word_index != 0:
                    word = self.vocab.get_token_from_index(word_index)
                    readable_seq.append(word)
            readable_sequences.append(readable_seq)
        return readable_sequences

    def compute_generator_loss(self,
                               embeddings: torch.Tensor,
                               code: torch.Tensor,
                               targets: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
        inputs = self.construct_generator_inputs(
            embeddings=embeddings, code=code)
        hiddens = self._generator(inputs.contiguous(), mask)
        logits = self._generator_projector(hiddens)
        weights = mask.float()
        loss = sequence_cross_entropy_with_logits(
            logits=logits, targets=targets.contiguous(), weights=weights,
            average=None)
        return loss

    def forward(self,
                premise: Dict[str, torch.Tensor],
                hypothesis: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None) -> Dict[str, Any]:
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

        pre_hidden = self.encode(
            inputs=pre_token_embs, mask=pre_mask, drop_start_token=True)
        hyp_hidden = self.encode(
            inputs=hyp_token_embs, mask=hyp_mask, drop_start_token=True)
        pre_code, pre_kld = self.sample_code_and_compute_kld(pre_hidden)
        hyp_code, hyp_kld = self.sample_code_and_compute_kld(hyp_hidden)
        pre_kld = pre_kld.mean()
        hyp_kld = hyp_kld.mean()

        pre_gen_mask = pre_mask[:, 1:]
        hyp_gen_mask = hyp_mask[:, 1:]
        pre_gen_loss = self.compute_generator_loss(
            embeddings=pre_token_embs[:, :-1], code=pre_code,
            targets=pre_tokens[:, 1:], mask=pre_gen_mask)
        hyp_gen_loss = self.compute_generator_loss(
            embeddings=hyp_token_embs[:, :-1], code=hyp_code,
            targets=hyp_tokens[:, 1:], mask=hyp_gen_mask)
        pre_gen_loss = pre_gen_loss.mean()
        hyp_gen_loss = hyp_gen_loss.mean()

        gen_loss = pre_gen_loss + hyp_gen_loss
        kld = pre_kld + hyp_kld
        loss = gen_loss + self.kl_weight*kld

        if label is not None:
            disc_logits = self.discriminate(premise_hidden=pre_code,
                                            hypothesis_hidden=hyp_code)
            disc_loss = functional.cross_entropy(
                input=disc_logits, target=label)
            loss = loss + self.discriminator_weight*disc_loss
            output_dict['discriminator_loss'] = disc_loss
            self._metrics['discriminator_loss'](disc_loss)
            self._metrics['discriminator_accuracy'](
                predictions=disc_logits, gold_labels=label)

        output_dict['generator_loss'] = gen_loss
        output_dict['kl_divergence'] = kld
        output_dict['loss'] = loss
        self._metrics['generator_loss'](gen_loss)
        self._metrics['kl_divergence'](kld)
        self._metrics['loss'](loss)

        return output_dict

    def get_metrics(self, reset: bool = False
                    ) -> Dict[str, Union[float, Dict[str, float]]]:
        metrics = {k: v.get_metric(reset=reset)
                   for k, v in self._metrics.items()}
        metrics['kl_weight'] = self.kl_weight
        metrics['discriminator_weight'] = self.discriminator_weight
        return metrics


def test_labeled():
    from pprint import pprint

    params = Params({
        'token_embedder': {
            'num_embeddings': 4,
            'embedding_dim': 3
        },
        'code_dist_type': 'vmf',
        'vmf_kappa': 100
    })
    vocab = Vocabulary()
    while True:
        vocab_size = vocab.get_vocab_size()
        if vocab_size == 4:
            break
        vocab.add_token_to_namespace('a' + str(vocab_size))
    model = SeparatedQuoraModel(params=params, vocab=vocab)
    premise = {'tokens': torch.randint(low=0, high=4, size=(5, 6))}
    hypothesis = {'tokens': torch.randint(low=0, high=4, size=(5, 7))}
    label = torch.randint(low=0, high=3, size=(5,))
    output = model(premise=premise, hypothesis=hypothesis, label=label)
    pprint(output)
    pprint(model.get_metrics())


def test_unlabeled():
    from pprint import pprint

    params = Params({
        'token_embedder': {
            'num_embeddings': 4,
            'embedding_dim': 3
        },
        'code_dist_type': 'gaussian'
    })
    vocab = Vocabulary()
    while True:
        vocab_size = vocab.get_vocab_size()
        if vocab_size == 4:
            break
        vocab.add_token_to_namespace('a' + str(vocab_size))
    model = SeparatedQuoraModel(params=params, vocab=vocab)
    premise = {'tokens': torch.randint(low=0, high=4, size=(5, 6))}
    hypothesis = {'tokens': torch.randint(low=0, high=4, size=(5, 7))}
    output = model(premise=premise, hypothesis=hypothesis, label=None)
    pprint(output)
    pprint(model.get_metrics())


if __name__ == '__main__':
    test_labeled()
    test_unlabeled()

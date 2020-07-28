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


class QuoraModel(Model):

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
        label_emb_dim = params.pop_int('label_emb_dim', 50)
        shared_encoder = params.pop_bool('shared_encoder', True)
        tie_embedding = params.pop_bool('tie_embedding', False)

        emb_dropout = params.pop_float('emb_dropout', 0.0)
        disc_dropout = params.pop_float('disc_dropout', 0.0)
        l2_weight = params.pop_float('l2_weight', 0.0)

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.disc_dropout = nn.Dropout(disc_dropout)
        self._l2_weight = l2_weight

        self._token_embedder = Embedding.from_params(
            vocab=vocab, params=params.pop('token_embedder'))
        self._label_embedder = Embedding(
            num_embeddings=self._NUM_LABELS,
            embedding_dim=label_emb_dim)
        self._encoder = PytorchSeq2VecWrapper(
            nn.LSTM(input_size=self._token_embedder.get_output_dim(),
                    hidden_size=enc_hidden_dim, batch_first=True))
        self._generator = PytorchSeq2SeqWrapper(
            nn.LSTM(input_size=(self._token_embedder.get_output_dim()
                                + code_dim
                                + label_emb_dim),
                    hidden_size=gen_hidden_dim, batch_first=True))
        self._generator_projector = nn.Linear(
            in_features=self._generator.get_output_dim(),
            out_features=vocab.get_vocab_size())
        self._discriminator_encoder = PytorchSeq2VecWrapper(
            nn.LSTM(input_size=self._token_embedder.get_output_dim(),
                    hidden_size=enc_hidden_dim, batch_first=True))
        if shared_encoder:
            self._discriminator_encoder = self._encoder
        if tie_embedding:
            self._generator_projector.weight = self._token_embedder.weight

        self._discriminator = FeedForward(
            input_dim=2 * self._discriminator_encoder.get_output_dim(),
            hidden_dims=[disc_hidden_dim]*disc_num_layers + [self._NUM_LABELS],
            num_layers=disc_num_layers + 1,
            activations=[Activation.by_name('relu')()] * disc_num_layers
                        + [Activation.by_name('linear')()],
            dropout=disc_dropout)
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
            raise ValueError('Unknown z_dist')

        self._kl_weight = 1.0
        self._discriminator_weight = params.pop_float(
            'discriminator_weight', 0.1)
        self._gumbel_temperature = 1.0

        # Metrics
        self._metrics = {
            'labeled': {
                'generator_loss': ScalarMetric(),
                'kl_divergence': ScalarMetric(),
                'discriminator_entropy': ScalarMetric(),
                'discriminator_accuracy': CategoricalAccuracy(),
                'discriminator_loss': ScalarMetric(),
                'loss': ScalarMetric()
            },
            'unlabeled': {
                'generator_loss': ScalarMetric(),
                'kl_divergence': ScalarMetric(),
                'discriminator_entropy': ScalarMetric(),
                'loss': ScalarMetric()
            },
            'aux': {
                'discriminator_entropy': ScalarMetric(),
                'discriminator_accuracy': CategoricalAccuracy(),
                'discriminator_loss': ScalarMetric(),
                'gumbel_temperature': ScalarMetric(),
                'loss': ScalarMetric(),
                'code_log_prob': ScalarMetric(),
                'cosine_dist': ScalarMetric()
             }
        }

    def add_finetune_parameters(self, con_autoweight=False,
                                con_y_weight=None, con_z_weight=None,
                                con_z2_weight=None):
        self.con_autoweight = con_autoweight
        self.con_y_weight = con_y_weight
        self.con_z_weight = con_z_weight
        self.con_z2_weight = con_z2_weight
        if con_autoweight:
            self.con_y_weight_p = nn.Parameter(torch.zeros(1))
            self.con_z_weight_p = nn.Parameter(torch.zeros(1))
            self.con_z2_weight_p = nn.Parameter(torch.zeros(1))

    def finetune_main_parameters(self, exclude_generator=False):
        params = []
        for name, param in self.named_parameters():
            if exclude_generator:
                if 'generator' in name:
                    continue
            params.append(param)
        return params

    def finetune_aux_parameters(self):
        gen_params = list(self._generator.parameters())
        gen_proj_params = list(self._generator_projector.parameters())
        emb_params = list(self._token_embedder.parameters())
        # enc_params = list(self._encoder.parameters())
        # code_gen_params = list(self._code_generator.parameters())
        con_params = []
        if self.con_autoweight:
            con_params = [self.con_y_weight_p, self.con_z_weight_p, self.con_z2_weight_p]
        return gen_params + gen_proj_params + emb_params + con_params

    def get_regularization_penalty(self):
        sum_sq = sum(p.pow(2).sum() for p in self.parameters())
        l2_norm = sum_sq.sqrt()
        return self.l2_weight * l2_norm

    @property
    def gumbel_temperature(self):
        return self._gumbel_temperature

    @gumbel_temperature.setter
    def gumbel_temperature(self, value):
        self._gumbel_temperature = value

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
            [premise_hidden + hypothesis_hidden,
             (premise_hidden - hypothesis_hidden).abs()],
            dim=-1)
        disc_input = self.disc_dropout(disc_input)
        disc_logits = self._discriminator(disc_input)
        return disc_logits

    def construct_generator_inputs(self,
                                   embeddings: torch.Tensor,
                                   code: torch.Tensor,
                                   label: torch.Tensor) -> torch.Tensor:
        batch_size, max_length, _ = embeddings.shape
        code_expand = code.unsqueeze(1).expand(
            batch_size, max_length, -1)
        label_emb = self._label_embedder(label)
        label_emb_expand = label_emb.unsqueeze(1).expand(
            batch_size, max_length, -1)
        inputs = torch.cat([embeddings, code_expand, label_emb_expand], dim=-1)
        return inputs

    def generate(self,
                 code: torch.Tensor,
                 label: torch.Tensor,
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
                embeddings=prev_word_emb, code=code, label=label)
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

    def gumbel_softmax(self,logits):
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        new_logits = (logits + g) / self.gumbel_temperature
        probs = functional.softmax(new_logits, dim=-1)
        return probs

    def generate_soft(self,
                      code: torch.Tensor,
                      label: torch.Tensor,
                      length: torch.Tensor) -> torch.Tensor:
        """
        Generate soft predictions using the Gumbel-Softmax
        reparameterization. Note that the generated sentence
        always has exactly the length of `length`.
        """
        start_index = self.vocab.get_token_index('<s>')
        batch_size = code.shape[0]
        prev_word = (torch.empty(batch_size, device=code.device).long()
                     .unsqueeze(1).fill_(start_index))
        prev_word_emb = self.embed(prev_word)
        max_length = length.max().item()
        generated_embs = []
        self._generator.stateful = True
        self._generator.reset_states()
        for t in range(max_length):
            input_t = self.construct_generator_inputs(
                embeddings=prev_word_emb, code=code, label=label)
            mask = length.gt(t).long().unsqueeze(1)
            hidden_t = self._generator(input_t, mask)
            logit_t = self._generator_projector(hidden_t)
            gumbel_probs_t = self.gumbel_softmax(logit_t)
            emb_t = torch.matmul(gumbel_probs_t, self._token_embedder.weight)
            generated_embs.append(emb_t)
            prev_word_emb = emb_t
        self._generator.stateful = False
        self._generator.reset_states()
        generated_embs = torch.cat(generated_embs, dim=1)
        return generated_embs

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
                               label: torch.Tensor,
                               targets: torch.Tensor,
                               mask: torch.Tensor,
                               ) -> torch.Tensor:
        inputs = self.construct_generator_inputs(
            embeddings=embeddings, code=code, label=label)
        hiddens = self._generator(inputs.contiguous(), mask)
        logits = self._generator_projector(hiddens)
        weights = mask.float()
        loss = sequence_cross_entropy_with_logits(
            logits=logits, targets=targets.contiguous(), weights=weights,
            average=None)
        return loss

    def aux_forward(self,
                    premise: Dict[str, torch.Tensor],
                    hypothesis: Dict[str, torch.Tensor]):
        """
        Generate the hypothesis dynamically given a premise
        and a sampled label, then compute the discriminator loss.
        This is intended to update only generator parameters,
        thus unnecessary gradients will not be accumulated
        for the reduced memory usage and faster training.
        """
        pre_mask = get_text_field_mask(premise)
        pre_tokens = premise['tokens']
        hyp_mask = get_text_field_mask(hypothesis)
        hyp_tokens = hypothesis['tokens']

        # with torch.no_grad():
        pre_token_embs = self.embed(pre_tokens)
        pre_hidden = self.encode(inputs=pre_token_embs, mask=pre_mask,
                                 drop_start_token=True)
        code, kld = self.sample_code_and_compute_kld(pre_hidden)
        batch_size = code.shape[0]
        label_dist = Categorical(
            logits=torch.ones(self._NUM_LABELS, device=code.device))
        label = label_dist.sample((batch_size,))

        gen_hyp_token_embs = self.generate_soft(
            code=code, label=label, length=pre_mask.sum(1))
        gen_hyp_hidden = self.encode(
            inputs=gen_hyp_token_embs, mask=pre_mask, drop_start_token=False)

        loss = 0
        output_dict = {}
        if self.con_y_weight > 0:
            disc_logits = self.discriminate(
                premise_hidden=pre_hidden,
                hypothesis_hidden=gen_hyp_hidden)
            disc_dist = Categorical(logits=disc_logits)
            disc_entropy = disc_dist.entropy().mean()
            disc_loss = functional.cross_entropy(
                input=disc_logits, target=label)
            if self.con_autoweight:
                yw = self.con_y_weight_p.exp().reciprocal()
                reg = self.con_y_weight_p * 0.5
                loss = loss + yw * disc_loss + reg
            else:
                loss = loss + self.con_y_weight * disc_loss
            output_dict['discriminator_entropy'] = disc_entropy
            output_dict['discriminator_loss'] = disc_loss
            self._metrics['aux']['discriminator_entropy'](disc_entropy)
            self._metrics['aux']['discriminator_loss'](disc_loss)
            self._metrics['aux']['discriminator_accuracy'](
                predictions=disc_logits, gold_labels=label)

        if self.con_z_weight > 0:
            hyp_token_embs = self.embed(hyp_tokens)
            hyp_hidden = self.encode(inputs=hyp_token_embs, mask=hyp_mask,
                                     drop_start_token=True)
            hyp_code, hyp_kld = self.sample_code_and_compute_kld(hyp_hidden)
            gen_hyp_dist = self._code_generator.get_distribution(
                gen_hyp_hidden)
            code_log_prob = -gen_hyp_dist.log_prob(hyp_code).mean()
            code_loss = -code_log_prob
            if self.con_autoweight:
                zw = self.con_z_weight_p.exp().reciprocal()
                reg = self.con_z_weight_p * 0.5
                loss = loss + zw * code_loss + reg
            else:
                loss = loss + self.con_z_weight * code_loss

            output_dict['code_loss'] = code_loss
            self._metrics['aux']['code_log_prob'](code_log_prob)

        if self.con_z2_weight > 0:
            gen_hyp_dist = self._code_generator.get_distribution(
                gen_hyp_hidden)
            mu = gen_hyp_dist.loc
            mu_bar = mu.mean(dim=0, keepdim=True)
            mu_bar = mu_bar / mu_bar.norm(dim=1, keepdim=True)
            cosine_dist = 1 - (mu * mu_bar).sum(dim=1)
            z2_loss = -cosine_dist.mean(dim=0)  # Scalar
            if self.con_autoweight:
                z2w = self.con_z2_weight_p.exp().reciprocal()
                reg = self.con_z2_weight_p * 0.5
                loss = loss + z2w * z2_loss + reg
            else:
                loss = loss + self.con_z2_weight * z2_loss
            output_dict['cosine_dist_mean'] = z2_loss
            self._metrics['aux']['cosine_dist'](-z2_loss)

        output_dict['loss'] = loss
        self._metrics['aux']['gumbel_temperature'](self.gumbel_temperature)
        self._metrics['aux']['loss'](loss)
        return output_dict

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

        if label is not None:  # Labeled
            pre_hidden = self.encode(
                inputs=pre_token_embs, mask=pre_mask, drop_start_token=True)
            # hyp_hidden = self.encode(
            #     inputs=hyp_token_embs, mask=hyp_mask, drop_start_token=True)
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

            code, kld = self.sample_code_and_compute_kld(pre_hidden)
            kld = kld.mean()
            gen_mask = hyp_mask[:, 1:]
            gen_loss = self.compute_generator_loss(
                embeddings=hyp_token_embs[:, :-1], code=code, label=label,
                targets=hyp_tokens[:, 1:], mask=gen_mask)
            gen_loss = gen_loss.mean()
            loss = (gen_loss + self.kl_weight*kld
                    + self.discriminator_weight*disc_loss)

            output_dict['discriminator_entropy'] = disc_entropy
            output_dict['discriminator_loss'] = disc_loss
            output_dict['generator_loss'] = gen_loss
            output_dict['kl_divergence'] = kld
            output_dict['loss'] = loss

            self._metrics['labeled']['discriminator_entropy'](disc_entropy)
            self._metrics['labeled']['discriminator_loss'](disc_loss)
            self._metrics['labeled']['discriminator_accuracy'](
                predictions=disc_logits, gold_labels=label)
            self._metrics['labeled']['generator_loss'](gen_loss)
            self._metrics['labeled']['kl_divergence'](kld)
            self._metrics['labeled']['loss'](loss)
        else:  # Unlabeled
            pre_hidden = self.encode(
                inputs=pre_token_embs, mask=pre_mask, drop_start_token=True)
            # hyp_hidden = self.encode(
            #     inputs=hyp_token_embs, mask=hyp_mask, drop_start_token=True)
            pre_disc_hidden = self.discriminator_encode(
                inputs=pre_token_embs, mask=pre_mask, drop_start_token=True)
            hyp_disc_hidden = self.discriminator_encode(
                inputs=hyp_token_embs, mask=hyp_mask, drop_start_token=True)
            disc_logits = self.discriminate(
                premise_hidden=pre_disc_hidden,
                hypothesis_hidden=hyp_disc_hidden)
            disc_dist = Categorical(logits=disc_logits)
            disc_entropy = disc_dist.entropy().mean()

            code, kld = self.sample_code_and_compute_kld(pre_hidden)
            kld = kld.mean()

            batch_size = pre_hidden.shape[0]
            label = torch.arange(self._NUM_LABELS, dtype=torch.long,
                                 device=pre_hidden.device)
            label_repeat = label.unsqueeze(1).repeat(1, batch_size).view(-1)
            targets_repeat = hyp_tokens[:, 1:].repeat(self._NUM_LABELS, 1)
            gen_mask_repeat = hyp_mask[:, 1:].repeat(self._NUM_LABELS, 1)
            hyp_token_embs_repeat = hyp_token_embs[:, :-1].repeat(
                self._NUM_LABELS, 1, 1)
            code_repeat = code.repeat(self._NUM_LABELS, 1)
            gen_loss = self.compute_generator_loss(
                embeddings=hyp_token_embs_repeat,
                code=code_repeat, label=label_repeat,
                targets=targets_repeat, mask=gen_mask_repeat)
            gen_loss = (gen_loss.contiguous().view(-1, self._NUM_LABELS)
                        * disc_dist.probs)
            gen_loss = gen_loss.sum(1).mean()
            loss = gen_loss + self.kl_weight*kld - disc_entropy

            output_dict['discriminator_entropy'] = disc_entropy
            output_dict['generator_loss'] = gen_loss
            output_dict['kl_divergence'] = kld
            output_dict['loss'] = loss

            self._metrics['unlabeled']['discriminator_entropy'](disc_entropy)
            self._metrics['unlabeled']['generator_loss'](gen_loss)
            self._metrics['unlabeled']['kl_divergence'](kld)
            self._metrics['unlabeled']['loss'](loss)

        return output_dict

    def get_metrics(self, reset: bool = False
                    ) -> Dict[str, Union[float, Dict[str, float]]]:
        metrics = {label_type: {k: v.get_metric(reset=reset)
                                for k, v in label_metrics.items()}
                   for label_type, label_metrics in self._metrics.items()}
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
    model = QuoraModel(params=params, vocab=vocab)
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
    model = QuoraModel(params=params, vocab=vocab)
    premise = {'tokens': torch.randint(low=0, high=4, size=(5, 6))}
    hypothesis = {'tokens': torch.randint(low=0, high=4, size=(5, 7))}
    output = model(premise=premise, hypothesis=hypothesis, label=None)
    pprint(output)
    pprint(model.get_metrics())


if __name__ == '__main__':
    test_labeled()
    test_unlabeled()

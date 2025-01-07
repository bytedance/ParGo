# Copyright (c) 2024 Bytedance Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import pargo.utils.disttools as disttools
from pargo.utils.metric import MetricLogger

from pargo.backbone.language.qformer_bert import (
    BertOnlyMLMHead,
    BertPreTrainedModel,
    BertModel,
    BertConfig,
    BertLMHeadModel,
)
from transformers.modeling_outputs import (
    MaskedLMOutput,
)


__all__ = ['BertForMaskedLM', 'init_ParGo', 'compute_sim_matrix']



class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def init_ParGo(bert_type, num_query_token, num_local_query_token, vision_width, cross_attention_freq=2):
    # bert_type == "bert-base-uncased"
    encoder_config = BertConfig.from_pretrained(bert_type)
    encoder_config.encoder_width = vision_width
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    encoder_config.local_query_length = num_local_query_token,
    ParGo = BertLMHeadModel.from_pretrained(bert_type, config=encoder_config,ignore_mismatched_sizes=True)
    query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return ParGo, query_tokens



def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = disttools.get_world_size()
    rank = disttools.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if disttools.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def compute_itm(model, image_inputs, text_ids, text_atts):
    image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
        image_inputs.device
    )
    query_tokens = model.query_tokens.expand(image_inputs.shape[0], -1, -1)
    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        image_inputs.device
    )
    attention_mask = torch.cat([query_atts, text_atts], dim=1)
    output_itm = model.ParGo.bert(
        text_ids,
        query_embeds=query_tokens,
        attention_mask=attention_mask,
        encoder_hidden_states=image_inputs,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
    itm_logit = model.itm_head(vl_embeddings)
    itm_logit = itm_logit[:, :, 1].mean(dim=1)
    return itm_logit


@torch.no_grad()
def generate(
    model,
    samples,
    use_nucleus_sampling=False,
    num_beams=3,
    max_length=30,
    min_length=10,
    top_p=0.9,
    repetition_penalty=1.0,
):
    """
    Args:
        samples (dict): A dictionary containing the following keys:
            - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
        use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
        num_beams (int): Number of beams for beam search. 1 means no beam search.
        max_length (int): The maximum length of the sequence to be generated.
        min_length (int): The minimum length of the sequence to be generated.
        top_p (float): The cumulative probability for nucleus sampling.
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        num_captions (int): Number of captions to be generated for each image.
    Returns:
        captions (list): A list of strings of length batch_size * num_captions.
    """
    image = samples["image"]
    image_embeds = model.visual_encoder(image)

    if not use_nucleus_sampling:
        image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
    else:
        num_beams = 1
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device
    )

    model_kwargs = {
        "encoder_hidden_states": image_embeds,
        "encoder_attention_mask": image_atts,
    }

    input_ids = (
        torch.LongTensor(image.size(0), 1)
        .fill_(model.tokenizer.bos_token_id)
        .to(image.device)
    )
    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)

    outputs = model.ParGo.generate(
        input_ids=input_ids,
        query_embeds=query_tokens,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=use_nucleus_sampling,
        top_p=top_p,
        eos_token_id=model.tokenizer.sep_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
        **model_kwargs
    )
    captions = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return captions

# Copyright (c) 2024 Bytedance Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
import jieba
from transformers import BertTokenizer, T5TokenizerFast, AutoTokenizer


__all__ = ['bertTokenizer', 't5TokenizerFast', 'autoTokenizerBlip', 'autoTokenizerLLM']


class TokenizerWrapper(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in jieba.cut(text, HMM=False):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def bertTokenizer(bert_type, use_jieba=False, do_lower_case=True, root="./", **kwargs):
    # root = kwargs.get('root', './')
    bert_type = os.path.join(root, bert_type)

    if use_jieba:
        tokenizer = TokenizerWrapper.from_pretrained(bert_type, do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_type, do_lower_case=do_lower_case)
        if not do_lower_case:
            upper_ab = list(map(chr, range(65, 91)))
            upper_case_vocs = ["##"+a for a in upper_ab] + upper_ab
            num_added_toks = tokenizer.add_tokens(upper_case_vocs)
            print(f"Adding {num_added_toks} upper cased vocabs.")
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def autoTokenizerBlip(bert_type, use_jieba=False, do_lower_case=True, **kwargs):
    root = kwargs.get('root', './')
    bert_type = os.path.join(root, bert_type)
    
    tokenizer = AutoTokenizer.from_pretrained(bert_type, do_lower_case=do_lower_case)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def t5TokenizerFast(t5_model, **kwargs):
    # t5 tokenizer originally support lower-cased words
    t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
    return t5_tokenizer


# def mt5Tokenizer(mt5_model='google/mt5-xl'):
#     mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_model)
#     return mt5_tokenizer

def autoTokenizerLLM(model='bigscience/bloomz-3b', **kwargs):
    tokenizer_params = {
        "use_fast": False,
        "trust_remote_code": True,
        **kwargs
    }
    tokenzier = AutoTokenizer.from_pretrained(model, **tokenizer_params)
    # tokenzier = AutoTokenizer.from_pretrained(model)
    return tokenzier

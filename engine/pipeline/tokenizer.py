from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import torch

@dataclass
class TokenizerConfig:
    vocab_size: int = 32000
    max_sequence_length: int = 2048
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1
    unk_token_id: int = 3 

class Tokenizer:
    def __init__(
        self,
        config: TokenizerConfig,
        vocab_file: Optional[Union[str, Path]] = None
    ):
        self.config = config

        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}

        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self._init_byte_vocab()
    def _init_byte_vocab(self):
        # Initialize special tokens first
        self.vocab["<pad>"] = self.config.pad_token_id
        self.vocab["<bos>"] = self.config.bos_token_id
        self.vocab["<eos>"] = self.config.eos_token_id
        self.vocab["<unk>"] = self.config.unk_token_id
        
        self.inv_vocab[self.config.pad_token_id] = "<pad>"
        self.inv_vocab[self.config.bos_token_id] = "<bos>"
        self.inv_vocab[self.config.eos_token_id] = "<eos>"
        self.inv_vocab[self.config.unk_token_id] = "<unk>"

        # Then add byte vocabulary
        for i in range(256):
            char = chr(i)
            if char not in self.vocab: 
                self.vocab[char] = i + 4  
                self.inv_vocab[i + 4] = char
        self.inv_vocab[self.config.eos_token_id] = "<eos>"

    
    def load_vocab(self, vocab_file: Union[str, Path]):
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
            self.vocab = vocab_data["vocab"]
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        if max_length is None:
            max_length = self.config.max_sequence_length

        tokens = list(text)
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab.get("", 3))

        if add_special_tokens:
            token_ids = [self.config.bos_token_id] + token_ids + [self.config.eos_token_id]

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        text = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {
                self.config.pad_token_id,
                self.config.bos_token_id,
                self.config.eos_token_id
            }:
                continue
            text.append(self.inv_vocab.get(token_id, ""))

        return "".join(text)
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> Dict[str, torch.Tensor]:
        if max_length is None:
            max_length = self.config.max_sequence_length

        batch_tokens = [
                self.encode(text, add_special_tokens, max_length)
                for text in texts
        ]

        batch_max_length = max(len(tokens) for tokens in batch_tokens)
        batch_max_length = min(batch_max_length, max_length)

        attnetion_mask = []
        padded_tokens = []

        for tokens in batch_tokens:
            padding_length = batch_max_length - len(tokens)

            if padding and padding_length > 0:
                tokens = tokens + [self.config.pad_token_id] * padding_length
                mask = [1] * (batch_max_length - padding_length) + [0] * padding_length
            else:
                tokens = tokens[:batch_max_length]
                mask = [1] * len(tokens)

            attention_mask.append(mask)
            padded_tokens.append(tokens)

        return {
            "input_ids": torch.tensor(padded_tokens),
            "attention_mask": torch.tensor(attention_mask)
        }
    def batch_decode(
        self,
        batch_token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        return [
            self.decode(token_ids.tolist(), skip_special_tokens)
            for token_ids in batch_token_ids
        ]



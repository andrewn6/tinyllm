import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Iterator
from dataclasses import dataclass
from ..compute.scheduler import Scheduler, OperationType
from ..models.transformer import Transformer, TransformerConfig
from .tokenizer import Tokenizer, TokenizerConfig

@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    use_cache: bool = True

class TextGenerator:
    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None
    ): 
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device or torch.device('cpu')

        self.model = self.model.to(self.device)

        self.current_sequence_id = 0

    def __call__(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None
    ) -> Union[str, List[str]]:
        config = config or GenerationConfig()

        if isinstance(prompt, str):
            return self.generate(prompt, config)

        return self.batch_generate(prompt, config)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        try:
            encoded = self.tokenizer.batch_encode(
                [prompt],
                add_special_tokens=True
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            output_ids = self._generate_tokens(
                    input_ids,
                    attention_mask,
                    config
            )

            return self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            raise Exception(f"Text generation failed: {str(e)}")
    
    def batch_generate(
        self,
        prompts: List[str],
        config: GenerationConfig
    ) -> List[str]:
        encoded = self.tokenizer.batch_encode(
            prompts,
            add_special_tokens=True,
            padding=True
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        output_ids = self._generate_tokens(
                input_ids,
                attention_mask,
                config
        )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    
    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        config = config or GenerationConfig()

        encoded = self.tokenizer.batch_encode(
            [prompt],
            add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        sequence_id = self.get_sequence_id()
        past_key_values = None

        for _ in range(config.max_new_tokens):
            logits, past_key_values = self.model(
                    input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    sequence_id=sequence_id,
                    use_cache=config.use_cache
            )

            next_token = self._sample_token(logits[:, -1, :], config)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device)
            ], dim=-1)



            token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
            if token_text:
                yield token_text


            if next_token.item() == self.tokenizer.config.eos_token_id:
                break

        self.model.clear_cache(sequence_id)


    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        sequence_id = self.get_sequence_id()
        past_key_values = None

        all_token_ids = input_ids

        for _ in range(config.max_new_tokens):
            logits, past_key_values = self.model(
                input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )

            next_tokens = self._sample_token(logits[:, -1, :], config)
            next_tokens = next_tokens.unsqueeze(-1)
            
            all_token_ids = torch.cat([all_token_ids, next_tokens], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=self.device)
            ], dim=1)
            
            input_ids = next_tokens

            if (next_tokens == self.tokenizer.config.eos_token_id).all():
                break

        return all_token_ids
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        if not config.do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)

        if config.temperature != 1.0:
            logits = logits / config.temperature

        if config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens
    
    def get_sequence_id(self) -> int:
        sequence_id = self.current_sequence_id
        self.current_sequence_id += 1

        return sequence_id
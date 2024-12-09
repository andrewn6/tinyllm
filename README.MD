# TinyLLM

Minimal, high-performance inference engine for LLM's -- used in development environments

## Overview
TinyLLM streamlines the inference pipeline with minimal overhead, focusing on memory efficiency and throughput optimization. We include a custom tokenizer for self-developed models, and it's compataibile with existing LLM's through our scheduling systme.

## Features
- Memory managment pruning
- Efficient batch processing and response streaming
- Optimized scheduling for multi-model deployments
- Custom tokenizer implmentation for self-developed models
- Seamless integration with existing LLM frameworks
- Inference API
- KV cache implementation
- Training CLI for development models
- Byte-level tokenization

*This is very much still an experiment, especially the tokenizer, our scheduler is somewhat well-written, memory management is decent.*

I'll continue to slowly improve these components over my weekends.

## Scope
This is solely a inference engine. It does not:
- Implement large model architectures
- Include pre-trained models
- Support distributed training

## How to use?

Clone repository
```
git clone https://github.com/andrewn6/tinyllm
```
```
pip install -e . 
```

Register your trained model
```
tinyllm model register transformer-19m v1 \
    --checkpoint models/tiny-19m.pt \
    --model-type native \
    --description "19M parameter transformer"
```

Serve and expose to localhost
```
tinyllm serve \
    --model-name mymodel \
    --port 8000 \
    --model-type native
```

List models
```
tinyllm model list
```



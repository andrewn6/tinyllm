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

*This is very much still an experiment, 

## Scope
This is solely a inference engine. It does not:
- Implement model architectures
- Provide training capabilities
- Include pre-trained models
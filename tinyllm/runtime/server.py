from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import torch
import time
import logging
import uvicorn
import asyncio

from ..metrics.prometheus import PrometheusMetrics, PrometheusConfig
from contextlib import asynccontextmanager

from ..compute.scheduler import Scheduler, OperationType, BatchConfig
from ..models.transformer import Transformer, TransformerConfig
from ..pipeline.tokenizer import Tokenizer, TokenizerConfig
from ..pipeline.generator import TextGenerator, GenerationConfig
from ..registry.registry import ModelRegistry

logger = logging.getLogger(__name__)
metrics = PrometheusMetrics(
    PrometheusConfig(
        port=int(os.getenv('METRICS_PORT', 8001)),
        enabled=os.getenv('ENABLE_METRICS', 'false').lower() == 'true'
    )
)

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True
    stream: Optional[bool] = False

class BatchGenerationRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True

class GenerationResponse(BaseModel):
    text: str
    tokens_generated: int
    time_taken: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TinyLLM server...")
    app.state.device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {app.state.device}")
    yield
    if hasattr(app.state, "generator"):
        del app.state.generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Shutting down TinyLLM server...")

app = FastAPI(
    title="TinyLLM",
    description="High-performance inference engine",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_generator():
    if not hasattr(app.state, "generator"):
        try:
            model_name = os.getenv("MODEL_NAME", "default")
            model_version = os.getenv("MODEL_VERSION")
            
            registry = ModelRegistry()
            model_info = registry.get_model(model_name, version=model_version)
            
            if not model_info:
                raise ValueError(f"Model {model_name} not found")
            
            # Load model from registry
            model_config = TransformerConfig(**model_info.config)
            model = Transformer(model_config)
            
            if model_info.checkpoint_path:
                state_dict = torch.load(
                    model_info.checkpoint_path, 
                    map_location=app.state.device
                )
                model.load_state_dict(state_dict)
            
            model.to(app.state.device)
            model.eval()
            
            # Initialize tokenizer
            tokenizer_config = TokenizerConfig(
                vocab_size=model_config.vocab_size,
                max_sequence_length=model_config.max_sequence_length
            )
            tokenizer = Tokenizer(tokenizer_config)

            # Initialize generator
            app.state.generator = TextGenerator(
                model=model,
                tokenizer=tokenizer,
                device=app.state.device
            )
            logger.info(f"Initialized model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "name": "TinyLLM Server",
        "status": "running",
        "device": str(app.state.device)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(app.state.device),
        "model_loaded": hasattr(app.state, "generator"),
        "metrics_enabled": metrics.config.enabled,
        "metrics_port": metrics.config.port if metrics.config.enabled else None
    }

@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    if not metrics.config.enabled:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    # Get current metrics
    return metrics.get_current_metrics()

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    initialize_generator()
    
    # Record request start
    metrics.record_request("generate")
    request_start = time.time()

    try:
        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample
        )

        if request.stream:
            # Record streaming request
            metrics.record_request("generate_stream")
            return StreamingResponse(
                app.state.generator.stream(request.prompt, config),
                media_type='text/event-stream'
            )

        # Generate text
        output = app.state.generator(request.prompt, config)
        
        # Calculate metrics
        tokens_generated = len(output.split()) - len(request.prompt.split())
        time_taken = time.time() - request_start

        # Record detailed metrics
        metrics.record_latency("generate", time_taken)
        metrics.record_tokens(len(request.prompt.split()), tokens_generated)
        metrics.record_throughput(tokens_generated / time_taken)
        
        # Record memory metrics if on CUDA
        if torch.cuda.is_available():
            metrics.record_gpu_memory(
                torch.cuda.memory_allocated(),
                torch.cuda.memory_reserved()
            )

        return GenerationResponse(
            text=output,
            tokens_generated=tokens_generated,
            time_taken=time_taken
        )

    except Exception as e:
        # Record error metrics with error type
        metrics.record_error(type(e).__name__)
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_batch")
async def generate_batch(request: BatchGenerationRequest):
    initialize_generator()
    
    # Record batch request metrics
    metrics.record_request("generate_batch")
    metrics.record_batch_size(len(request.prompts))
    request_start = time.time()

    try:
        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        
        outputs = app.state.generator(request.prompts, config)
        time_taken = time.time() - request_start
        
        results = []
        total_input_tokens = 0
        total_generated_tokens = 0
        
        for prompt, output in zip(request.prompts, outputs):
            tokens_generated = len(output.split()) - len(prompt.split())
            total_input_tokens += len(prompt.split())
            total_generated_tokens += tokens_generated
            results.append({
                "text": output,
                "tokens_generated": tokens_generated
            })
        
        # Record detailed batch metrics
        metrics.record_latency("generate_batch", time_taken)
        metrics.record_tokens(total_input_tokens, total_generated_tokens)
        metrics.record_throughput(total_generated_tokens / time_taken)
        metrics.record_batch_latency_per_sequence(time_taken / len(request.prompts))
        
        # Record memory metrics if on CUDA
        if torch.cuda.is_available():
            metrics.record_gpu_memory(
                torch.cuda.memory_allocated(),
                torch.cuda.memory_reserved()
            )
            
        return {
            "results": results,
            "time_taken": time_taken
        }
        
    except Exception as e:
        metrics.record_error(type(e).__name__)
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get detailed server statistics including metrics"""
    stats = {
        "device": str(app.state.device),
        "model_loaded": hasattr(app.state, "generator"),
        "metrics_enabled": metrics.config.enabled
    }
    
    if app.state.device.type == "cuda":
        stats["memory"] = {
            "allocated": torch.cuda.memory_allocated(app.state.device),
            "reserved": torch.cuda.memory_reserved(app.state.device),
            "max_allocated": torch.cuda.max_memory_allocated(app.state.device)
        }
    
    if metrics.config.enabled:
        stats["metrics"] = metrics.get_summary()
    
    return stats

@app.on_event("startup")
async def startup():
    if metrics.config.enabled:
        from prometheus_client import start_http_server
        start_http_server(metrics.config.port)
        logger.info(f"Started Prometheus metrics server on port {metrics.config.port}")

    model_name = os.getenv('MODEL_NAME')
    model_version = os.getenv('MODEL_VERSION')

    if not model_name:
        raise ValueError("MODEL_NAME environment variable must be set")
    
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")

    app.state.scheduler = Scheduler(
        batch_config=BatchConfig(
            max_batch_size=32,
            max_sequence_length=128,
            dynamic_batching=True
        ),
        device=app.state.device
    )

    registry = ModelRegistry()
    model_info = registry.get_model(model_name, model_version)

    if not model_info:
        available = registry.list_models()
        raise ValueError(f"Model {model_name} not found.")
    
    logger.info(f"Loading model {model_name} from {model_info.checkpoint_path}")

    """Initialize model components"""
    model_config = TransformerConfig(**model_info.config)
    model = Transformer(model_config)
    model.load_state_dict(torch.load(model_info.checkpoint_path, map_location=app.state.device))
    model = model.to(app.state.device)
    model.eval()

    tokenizer_config = TokenizerConfig(
        vocab_size=model_config.vocab_size,
        max_sequence_length=model_config.max_sequence_length
    )
    tokenizer = Tokenizer(tokenizer_config)

    app.state.generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=app.state.device
    )

    if metrics.config.enabled:
        async def update_metrics():
            while True:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_used = torch.cuda.memory_allocated(i)
                        memory_reserved = torch.cuda.memory_reserved(i)
                        metrics.update_gpu_memory(f"cuda:{i}", memory_used)
                        metrics.update_gpu_reserved_memory(f"cuda:{i}", memory_reserved)
                    
                if hasattr(app.state, "generator"):
                    metrics.update_model_stats(app.state.generator)
                
                await asyncio.sleep(15)
                
        asyncio.create_task(update_metrics())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from ..ollama.adapter import OllamaAdapter
import torch
import time
import logging
import uvicorn
import asyncio

from ..metrics.prometheus import PrometheusMetrics, PrometheusConfig
from contextlib import asynccontextmanager
from ..models.transformer import Transformer, TransformerConfig
from ..pipeline.tokenizer import Tokenizer, TokenizerConfig
from ..pipeline.generator import TextGenerator, GenerationConfig

logger = logging.getLogger(__name__)
metrics = PrometheusMetrics(PrometheusConfig())

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
            "mps" if torch.backends.mps.is_available() else
            "cpu"
    )
    logger.info("Shutting down FastLLM server...")
    yield
    if hasattr(app.state, "generator"):
        del app.state.generator
        if torch.cuda.is_available():
            torch.cuda.empty_cach()


app = FastAPI(
    title="TinyLLM",
    description="Inference at warp-speed",
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
            model_type = os.getenv("MODEL_TYPE", "native")
            model_name = os.getenv("MODEL_NAME", "default")

            if model_type == "ollama":
                app.state.generator = OllamaAdapter(
                    model_name,
                    device=app.state.device
                )
                logger.info(f"Initialized Ollama model: {model_name}")
                return
            
            hidden_size = 1024  
            num_heads = 16
            
            model_config = TransformerConfig(
                num_layers=8,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_sequence_length=2048,
                vocab_size=32000,
                dropout=0.1,
                layer_norm_epsilon=1e-5,
                use_cache=True
            )
            
            tokenizer_config = TokenizerConfig(
                vocab_size=model_config.vocab_size,
                max_sequence_length=model_config.max_sequence_length
            )

            model = Transformer(model_config)
            if app.state.device.type == "cuda":
                torch.cuda.empty_cache()
            model.to(app.state.device)
            tokenizer = Tokenizer(tokenizer_config)

            app.state.generator = TextGenerator(
                model=model,
                tokenizer=tokenizer,
                device=app.state.device
            )
            logger.info("Generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize model")

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
        "model_loaded": hasattr(app.state, "generator")
    }

@app.post('/generate', response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    initialize_generator()

    try: 
        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample
        )

        start_time = time.time()

        if request.stream:
            return StreamingResponse(
                    app.state.generator.stream(request.prompt, config),
                    media_type='text/event-stream'
            )

        output = app.state.generator(request.prompt, config)
        tokens_generated = len(output.split()) - len(request.prompt.split())
        time_taken = time.time() - start_time

        return GenerationResponse(
                text=output,
                tokens_generated=tokens_generated,
                time_taken=time_taken
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_batch")
async def generate_batch(request: BatchGenerationRequest):
    initialize_generator()

    try:
        config = GenerationConfig(  # Make sure this line is properly indented
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        
        start_time = time.time()  # This line should be at the same level as config
        outputs = app.state.generator(request.prompts, config)
        time_taken = time.time() - start_time
        
        results = []
        for prompt, output in zip(request.prompts, outputs):
            tokens_generated = len(output.split()) - len(prompt.split())
            results.append({
                "text": output,
                "tokens_generated": tokens_generated
            })
            
        return {
            "results": results,
            "time_taken": time_taken
        }
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    stats = {
        "device": str(app.state.device),
        "model_loaded": hasattr(app.state, "generator")
    }
    
    # Add device-specific stats
    if app.state.device.type == "cuda":
        stats["memory"] = {
            "allocated": torch.cuda.memory_allocated(app.state.device),
            "reserved": torch.cuda.memory_reserved(app.state.device),
            "max_allocated": torch.cuda.max_memory_allocated(app.state.device)
        }
    
    return stats


@app.post("/generate")
async def generate_text(request: GenerationRequest):
    start_time = time.time()
    try:
        metrics.record_request("transformer", "generate")
        
        output = app.state.generator(
            request.prompt,
            config=GenerationConfig(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=request.do_sample
            )
        )
        
        duration = time.time() - start_time
        metrics.record_latency("transformer", "generate", duration)
        metrics.record_tokens(
            "transformer",
            len(request.prompt.split()),
            len(output.split())
        )
        
        return {"text": output}
        
    except Exception as e:
        metrics.record_error("transformer", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup():
    async def update_gpu_metrics():
        while True:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(i)
                    metrics.update_gpu_memory(f"cuda:{i}", memory_used)
            await asyncio.sleep(15)
    
    asyncio.create_task(update_gpu_metrics())

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
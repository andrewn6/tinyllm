import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from engine.models.tiny import TinyLLM
import json
import os
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_tiny(
    input_file: str = os.path.join(os.path.dirname(__file__), "input.txt"),
    output_dir: str = "checkpoints/tiny",
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    num_epochs: int = 10,
    save_every: int = 1000,
    device: str = get_device()
):
    print(f"Using device: {device}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Length of dataset in characters: {len(text):,}")

    model = TinyLLM().to(device)
    
    encoded = model.tokenizer.batch_encode(
        [text[i:i+128] for i in range(0, len(text)-128, batch_size)],
        padding=True
    )
    input_ids = encoded["input_ids"]  
    
    n = len(input_ids)
    train_data = input_ids[:int(n*0.9)]
    val_data = input_ids[int(n*0.9):]
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model.config.__dict__, f, indent=2)
        
    print(f"Training TinyLLM on {device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    train_dataloader = train_data
    val_dataloader = val_data
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            input_ids = batch.to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            
            optimizer.zero_grad()
            loss.backward()

            if device == "mps":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()
            
            loss_value = loss.detach().cpu().item()
            
            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
            
            if global_step % save_every == 0:
                checkpoint_path = os.path.join(output_dir, f"step_{global_step}.pt")
                if device == "mps":
                    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    state_dict = model.state_dict()
                    
                torch.save({
                    "step": global_step,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_value
                }, checkpoint_path)
                
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch.to(device)
                outputs = model(input_ids)
                val_loss += criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    input_ids.view(-1)
                ).item()
        val_loss /= len(val_dataloader)
        print(f"Validation loss: {val_loss:.4f}")
    
    final_path = os.path.join(output_dir, "final.pt")
    if device == "mps":
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, final_path)
    
    print("\nTraining complete!")
    print(f"Model saved to: {final_path}")
    print(f"Config saved to: {config_path}")
    print("\nTo register with engine:")
    print(f"engine model register tiny 1.0.0 \\")
    print(f"  --config {config_path} \\")
    print(f"  --checkpoint {final_path}")

if __name__ == "__main__":
    train_tiny()
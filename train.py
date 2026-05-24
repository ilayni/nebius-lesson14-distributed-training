import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_ID      = os.environ.get("MODEL_ID",                     "facebook/opt-2.7b")
BLOCK_SIZE    = int(os.environ.get("BLOCK_SIZE",               "512"))
BATCH_SIZE    = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "4"))
GRAD_ACCUM    = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1"))
NUM_WORKERS   = int(os.environ.get("DATALOADER_NUM_WORKERS",    "4"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS",                 "100"))
LOG_INTERVAL  = int(os.environ.get("LOG_INTERVAL",             "10"))


def tokenize(examples, tokenizer):
    tokens = tokenizer(examples["text"], truncation=True, max_length=BLOCK_SIZE, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"World size: {world_size}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset = dataset.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    sampler    = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    step          = 0
    running_loss  = 0.0

    for epoch in range(9999):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss / GRAD_ACCUM
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % LOG_INTERVAL == 0 and rank == 0:
                print(f"[rank {rank}] step {step}/{MAX_STEPS}  loss={running_loss / LOG_INTERVAL:.4f}", flush=True)
                running_loss = 0.0

            if step >= MAX_STEPS:
                break
        if step >= MAX_STEPS:
            break

    if rank == 0:
        print(f"Training complete — {step} steps.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

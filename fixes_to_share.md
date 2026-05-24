# Fixes & Gotchas — Nebius DDP Assignment

## 1. Nebius Container Registry push URL — strip `registry-` prefix

The Docker push URL does **not** use the full registry ID. Strip the `registry-` prefix:

```bash
# Full registry ID from CLI: registry-e00xxxxxxxxxxxx
# Correct push path:         e00xxxxxxxxxxxx

REG=cr.eu-north1.nebius.cloud/$(nebius registry list --parent-id <project-id> --format json \
  | jq -r '.items[] | select(.metadata.name=="<registry-name>") | .metadata.id | split("-")[1]')

docker push $REG/my-image:v1
```

Using the full ID (e.g. `cr.eu-north1.nebius.cloud/registry-e00xxx/...`) gives:
`error from registry: Entity Registry not found by id registry-e00xxx`

## 2. `kubernetes` Python client v36 breaks SkyPilot auth

`kubernetes==36.0.0` (latest) breaks bearer-token auth and `sky launch`. Use `29.0.0`:

```bash
pip install "kubernetes==29.0.0"
```

**Error with v36:**
```
401 Unauthorized — Bearer token never sent in the request
```

## 3. `wikitext` dataset renamed on HuggingFace Hub

**Error:**
```
huggingface_hub.errors.HfUriError: Invalid HF URI '...'. Repository id must be 'namespace/name', got 'wikitext'.
```

**Fix:** The dataset moved to `Salesforce/wikitext`. Update your `train.py` and `train_cpu.py`:

```python
# Before (broken)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# After (working)
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
```

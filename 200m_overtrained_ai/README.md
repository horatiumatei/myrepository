# 200M LLM — CulturaX Romanian (25:1 Ratio)

Model de limbaj LLaMA-style cu **~210 milioane de parametri**, antrenat pe **18.5GB de text românesc** din datasetul [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX).

> **25:1 Ratio** = Modelul este antrenat pe ~5.25 miliarde de tokeni (25 tokeni pentru fiecare parametru). Aceasta reprezinta o eficienta foarte buna a resurselor de antrenare vs calitatea inferentei.

---

## Arhitectura

| Parametru | Valoare |
|-----------|---------|
| Parametri totali | **~210M** |
| Layers | 14 |
| Dimensiune embeddings | 1024 |
| Attention heads | 16 |
| FFN inner dim | 4096 |
| Context length | 2048 tokens |
| Tokenizer | ByteLevel BPE, 32k vocab (antrenat pe română) |

---

## Utilizare în Google Colab (L4 GPU)

### Pasul 0 — Configurare inițială (o singură dată)

```python
# Celula 1: Instalare dependențe
!pip install -q torch transformers datasets tokenizers accelerate wandb pyyaml huggingface_hub ninja
!pip install flash-attn --no-build-isolation -q

# Celula 2: Autentificare GCP și montare bucket GCS
from google.colab import auth
auth.authenticate_user()

# Instalare gcsfuse
!echo "deb https://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
!apt -qq update
!apt -qq install gcsfuse

# Montare (inlocuieste <NUME_BUCKET> cu bucket-ul tau)
!mkdir -p /content/gcs
!gcsfuse <NUME_BUCKET> /content/gcs

# Celula 3: Clone repo + cd
!git clone https://github.com/USERUL/200m_overtrained_ai.git
%cd 200m_overtrained_ai

# Celula 4: Autentificare HuggingFace (necesara pentru CulturaX)
from huggingface_hub import login
login()  # introduce token-ul din https://huggingface.co/settings/tokens
```

---

### Pasul 1 — Antrenare tokenizer BPE (o singură dată, ~1-2 ore)

```python
# Celula 5: Antrenare tokenizer pe 2GB CulturaX Romanian
!python src/tokenizer_train.py --config config/train_config.yaml

# Dupa terminare, tokenizer-ul e la: ./tokenizer/ro_bpe_32k/
# Copiaza-l in GCP pentru a nu-l pierde la resetarea Colab:
!cp -r ./tokenizer/ro_bpe_32k /content/gcs/200m_ro_tokenizer
```

> **Tokenizer custom vs GPT-2 tokenizer**: Tokenizer-ul custom este superior pentru română —  
> GPT-2 fragmentează cuvintele românești ineficient (ex: "mâncare" → 5+ tokens vs 2-3 cu BPE roman).

---

### Pasul 2 — Antrenare model (~30 ore pe L4, multi-sesiune)

```python
# Celula 6: Copiaza tokenizer-ul din GCP (daca e o noua sesiune Colab)
!cp -r /content/gcs/200m_ro_tokenizer ./tokenizer/ro_bpe_32k

# Celula 7: START / RESUME antrenare
# Scriptul detecteaza automat ultimul checkpoint din GCP si reia de acolo
!python src/train.py \
    --model_config config/model_config.yaml \
    --train_config config/train_config.yaml
```

**Checkpoints**: salvate automat la `/content/gcs/200m_ro_checkpoints/` la fiecare 1000 de steps.  
**Resume**: la fiecare nouă sesiune Colab, montează iar bucket-ul și scriptul reia automat din ultimul checkpoint.

---

### Pasul 3 — Inferență / Test

```python
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

model_path = "/content/gcs/200m_ro_checkpoints/final_model"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).cuda()
model.eval()

prompt = "România este o țară"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=1.1,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Estimări de timp (L4 GPU Colab)

| Etapă | Timp estimat |
|-------|-------------|
| Antrenare tokenizer (2GB) | ~1-2 ore |
| Antrenare model complet (18.5GB / 5.25B tokens) | ~30 ore (2-3 reporniri sesiune) |
| Tokens/secundă pe L4 | ~20,000-30,000 tok/s |

> Sesiunile Colab durează max 12-24h. Scriptul reia automat datorită checkpointingului pe Drive.

---

## Structura proiectului

```
200m_overtrained_ai/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── model_config.yaml    # Arhitectura ~210M params
│   └── train_config.yaml    # Hyperparameteri antrenare
├── src/
│   ├── utils.py             # Logging, config, checkpointing
│   ├── tokenizer_train.py   # Antrenare tokenizer BPE
│   ├── dataset.py           # Streaming CulturaX + packing
│   ├── model.py             # GPT2 model config & init
│   └── train.py             # Loop antrenare principal
└── tokenizer/               # (generat la Pasul 1)
    └── ro_bpe_32k/
```

---

## Cerințe

- Python 3.9+
- CUDA GPU (recomandat: L4 24GB, A100, RTX 4090)
- Cont HuggingFace cu acces la [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)
- Bucket Google Cloud Storage (GCP) pentru persistența checkpointurilor

---

## Referințe

- [CulturaX Dataset](https://huggingface.co/datasets/uonlp/CulturaX)
- [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [GPT-2 Architecture (OpenAI)](https://huggingface.co/docs/transformers/model_doc/gpt2)

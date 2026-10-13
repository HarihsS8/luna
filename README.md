# LUNA: Language Understanding Neural Attention

**LUNA** is a custom **BERT-like bidirectional Transformer architecture** implemented in Python (PyTorch) for joint multi-task **sentiment classification** and **toxic comment detection**, paired with a clean, minimalist web interface and an Express API.

---

## Architecture Overview

Unlike causal autoregressive models (such as GPT), **LUNA** uses bidirectional self-attention to condition on both left and right context tokens across every layer.

```
Input Sentence
      │
      ▼
WordPiece Tokenizer ([CLS] + tokens + [SEP])
      │
      ▼
LunaEmbeddings (Token + Positional + Segment Embeddings + LayerNorm)
      │
      ▼
LunaEncoder Stack (6 Transformer Layers × 8 Multi-Head Attention Heads)
      │
      ├─── Scaled Dot-Product Attention: Softmax((Q · K^T) / √d_k + M) · V
      ├─── Add & LayerNorm (Residual Connection)
      ├─── Position-wise Feed-Forward Network: GELU(x · W_1 + b_1) · W_2 + b_2
      └─── Add & LayerNorm (Residual Connection)
      │
      ▼
LunaPooler ([CLS] Representation via Linear + Tanh)
      │
      ├─────────────────────────────────────────────┐
      ▼                                             ▼
Sentiment Classification Head           Toxic Comment Multi-Label Head
(Negative · Neutral · Positive)         (6 Binary Sigmoid Classifiers)
Loss: CrossEntropyLoss                  Loss: BCEWithLogitsLoss
```

### Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | 30,522 | WordPiece subword vocabulary size |
| `hidden_size` | 256 | Hidden representation dimension ($d_{\text{model}}$) |
| `num_hidden_layers` | 6 | Number of transformer encoder blocks |
| `num_attention_heads` | 8 | Attention heads ($d_k = 32$ per head) |
| `intermediate_size` | 1,024 | Pointwise FFN dimension ($4 \times d_{\text{model}}$) |
| `hidden_act` | `gelu` | Gaussian Error Linear Unit activation |
| `max_position_embeddings` | 512 | Maximum sequence length |
| `hidden_dropout_prob` | 0.1 | Dropout regularization probability |

---

## Multi-Task Classification Heads

LUNA jointly trains on two distinct natural language understanding tasks sharing a single encoder backbone:

### 1. Sentiment Classification Head
- **Target Classes**: 3 (`Negative`, `Neutral`, `Positive`)
- **Activation**: Softmax over 3 logits
- **Loss Function**: Multi-class Cross-Entropy Loss
$$\mathcal{L}_{\text{sentiment}} = -\sum_{c=1}^{3} y_c \log(\hat{y}_c)$$

### 2. Toxic Comment Detection Head
- **Target Categories**: 6 independent multi-label dimensions:
  1. `toxic` — General toxicity, profanity, and hostile remarks
  2. `severe_toxic` — Aggressive, destructive vitriol
  3. `obscene` — Vulgarities, swear words, and explicit terms
  4. `threat` — Expressions of violence or intent to inflict harm
  5. `insult` — Inflammatory or demeaning personal attacks
  6. `identity_hate` — Hateful attacks targeting protected groups
- **Activation**: Independent Sigmoids ($\sigma(z_j) \in [0, 1]$)
- **Loss Function**: Binary Cross-Entropy with Logits
$$\mathcal{L}_{\text{toxic}} = -\frac{1}{6} \sum_{j=1}^{6} \left[ y_j \log(\sigma(z_j)) + (1 - y_j) \log(1 - \sigma(z_j)) \right]$$

### Joint Loss
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{sentiment}} + \beta \mathcal{L}_{\text{toxic}}$$

---

## Project Structure

```
├── luna/                          # Core Python PyTorch package
│   ├── __init__.py                # Package exports
│   ├── config.py                  # LunaConfig hyperparameter dataclass
│   ├── tokenizer.py               # Subword WordPiece tokenizer
│   ├── model.py                   # Bidirectional Transformer backbone
│   ├── heads.py                   # Sentiment and Toxic classification heads
│   ├── dataset.py                 # PyTorch Dataset and synthetic benchmark
│   ├── train.py                   # AdamW training loop with metric logging
│   └── eval.py                    # Inference pipeline and attribution
├── src/                           # Frontend React application
│   ├── components/                # Minimalist UI components
│   │   ├── TopBar.tsx             # Clean navigation header
│   │   └── InferenceLab.tsx       # Text classification interface
│   ├── services/                  # Inference engine and API client
│   │   ├── lunaEngine.ts          # Client-side BERT transformer forward pass
│   │   ├── pythonFiles.ts         # Bundled Python source modules
│   │   └── exportZip.ts           # Zip export utility
│   ├── types/                     # TypeScript data interfaces
│   ├── App.tsx                    # Main application root
│   └── index.css                  # Tailwind CSS styling
├── server.ts                      # Full-stack Express backend & API
├── quickstart.py                  # Standalone Python verification script
├── requirements.txt               # Python package dependencies
├── package.json                   # Node.js dependencies and scripts
├── tsconfig.json                  # TypeScript compiler configuration
└── vite.config.ts                 # Vite bundler configuration
```

---

## Getting Started

### 1. Python Environment (PyTorch)

```bash
# Clone the repository
git clone https://github.com/your-username/luna.git
cd luna

# Install Python dependencies
pip install -r requirements.txt

# Run the quickstart verification script
python quickstart.py
```

#### Running Training
```bash
python -m luna.train
```

#### Python Example Usage
```python
from luna import LunaConfig, LunaTokenizer, LunaForMultiTaskClassification
from luna.eval import LunaInferencePipeline

# Initialize model and tokenizer
config = LunaConfig(hidden_size=256, num_hidden_layers=6, num_attention_heads=8)
tokenizer = LunaTokenizer()
model = LunaForMultiTaskClassification(config)

# Run inference
pipeline = LunaInferencePipeline(model, tokenizer)
result = pipeline.predict("The camera build is stellar, but battery life was terrible.")

print(result["sentiment"]["label"])       # Output: "Negative"
print(result["toxicity"]["is_toxic"])     # Output: False
```

---

### 2. Full-Stack Web Application

The interactive web UI provides real-time forward pass analysis, word attribution saliency, and a self-attention matrix inspector.

```bash
# Install Node dependencies
npm install

# Start the development server (runs Express on port 3000 with Vite)
npm run dev

# Build for production
npm run build
```

The web application will be accessible at `http://localhost:3000`.

---

## API Endpoints

### `POST /api/classify`
Classifies input text for sentiment polarity and toxic sub-labels.

**Request Body:**
```json
{
  "text": "The camera build is stellar and customer service was fantastic!"
}
```

**Response:**
```json
{
  "text": "The camera build is stellar and customer service was fantastic!",
  "tokens": ["[CLS]", "the", "camera", "build", "is", "stellar", "and", "customer", "service", "was", "fantastic", "!", "[SEP]"],
  "sentiment": {
    "label": "positive",
    "score": 0.942,
    "probabilities": {
      "positive": 0.942,
      "neutral": 0.048,
      "negative": 0.010
    }
  },
  "toxicity": {
    "overallRisk": "clean",
    "overallScore": 0.012,
    "categories": [
      { "name": "General Toxicity", "probability": 0.012, "isFlagged": false, "threshold": 0.5 },
      { "name": "Severe Toxicity", "probability": 0.001, "isFlagged": false, "threshold": 0.4 },
      { "name": "Insult", "probability": 0.008, "isFlagged": false, "threshold": 0.45 },
      { "name": "Threat of Violence", "probability": 0.002, "isFlagged": false, "threshold": 0.35 },
      { "name": "Obscenity & Vulgarity", "probability": 0.005, "isFlagged": false, "threshold": 0.5 },
      { "name": "Identity Attack", "probability": 0.002, "isFlagged": false, "threshold": 0.4 }
    ]
  },
  "inferenceTimeMs": 1.8
}
```

### `GET /api/model/config`
Returns model hyperparameters, token dimensions, and loss functions.

---

## License

This project is licensed under the Apache 2.0 License.

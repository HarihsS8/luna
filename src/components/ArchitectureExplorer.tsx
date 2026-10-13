import React, { useState } from 'react';
import { 
  Network, 
  Layers, 
  Code, 
  Copy, 
  Check, 
  ArrowDown, 
  Binary, 
  Split, 
  Cpu, 
  Sigma,
  GitBranch
} from 'lucide-react';
import { ArchitectureBlock } from '../types/luna';

const ARCHITECTURE_BLOCKS: ArchitectureBlock[] = [
  {
    id: 'tokenizer',
    name: 'WordPiece Subword Tokenizer',
    subTitle: 'Input preprocessing & special token injection',
    category: 'input',
    tensorShape: 'Text String → Int IDs [batch, seq_len]',
    mathFormula: 'tokens = [CLS] + WordPiece(text) + [SEP] + [PAD]*k',
    description: 'Converts unstructured raw sentences into token sequences, decomposing out-of-vocabulary words into frequent subword prefixes with ## markers. Injects [CLS] at position 0 to act as sentence-level aggregate vector.',
    pythonSnippet: `tokenizer = LunaTokenizer()
encoded = tokenizer.encode(
    "The camera build is stellar, but support was bad!",
    max_length=128,
    padding=True,
    truncation=True
)
# Returns: input_ids, attention_mask, token_type_ids`
  },
  {
    id: 'embeddings',
    name: 'LunaEmbeddings Layer',
    subTitle: 'Tri-part vector summation with LayerNorm',
    category: 'embedding',
    tensorShape: '[batch, seq_len] → [batch, seq_len, hidden_size=256]',
    mathFormula: 'E = LayerNorm(E_token(x) + E_pos(p) + E_segment(s))',
    description: 'Combines word identity, spatial absolute positional order, and segment sentence type embeddings into a unified dense 256-dimensional space, normalized with LayerNorm and regularized with dropout.',
    pythonSnippet: `class LunaEmbeddings(nn.Module):
    def __init__(self, config: LunaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)`
  },
  {
    id: 'self_attention',
    name: 'Multi-Head Self-Attention',
    subTitle: 'Bidirectional context aggregation across 8 heads',
    category: 'encoder',
    tensorShape: '[batch, seq_len, 256] → [batch, 8, seq_len, 32]',
    mathFormula: 'Attention(Q, K, V) = Softmax((Q · K^T) / √d_k + M) · V',
    description: 'Allows every token to attend simultaneously to all tokens to its left and right without causal masking. Splits the 256-d vector into 8 parallel 32-d heads, each capturing distinct syntactic or semantic relationships (modifiers, insults, punctuation, negation).',
    pythonSnippet: `class LunaSelfAttention(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = nn.functional.softmax(scores, dim=-1)
        return torch.matmul(probs, value)`
  },
  {
    id: 'ffn',
    name: 'Feed-Forward Network (FFN)',
    subTitle: 'Pointwise intermediate expansion with GELU',
    category: 'encoder',
    tensorShape: '[batch, seq_len, 256] → [batch, seq_len, 1024] → [batch, seq_len, 256]',
    mathFormula: 'FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2',
    description: 'Expands representation by 4x to intermediate dimension (1024) using Gaussian Error Linear Unit (GELU) non-linearity before projecting back to 256-d, followed by residual addition and layer normalization.',
    pythonSnippet: `class LunaIntermediate(nn.Module):
    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # 256 -> 1024
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.act_fn(self.dense(x))`
  },
  {
    id: 'pooler',
    name: 'LunaPooler ([CLS] Aggregator)',
    subTitle: 'Extracts sentence-level summary vector',
    category: 'pooler',
    tensorShape: '[batch, seq_len, 256] → [batch, 256]',
    mathFormula: 'h_[CLS] = tanh(W_pool · H_0 + b_pool)',
    description: 'Extracts the first token hidden state H[0] representing the special [CLS] marker which has attended to all words across the 6 layers, transformed through a Linear layer and Tanh activation.',
    pythonSnippet: `class LunaPooler(nn.Module):
    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token = hidden_states[:, 0]  # [CLS] token
        return self.activation(self.dense(first_token))`
  },
  {
    id: 'sentiment_head',
    name: 'Sentiment Classification Head',
    subTitle: '3-class softmax polarity decision',
    category: 'heads',
    tensorShape: '[batch, 256] → [batch, 3]',
    mathFormula: 'P(y = c) = exp(z_c) / Σ exp(z_j)  |  Loss: CrossEntropy',
    description: 'Projects the 256-d pooled vector through dropout and a linear classifier to produce logits for [Negative, Neutral, Positive]. Optimized via Cross-Entropy Loss.',
    pythonSnippet: `class LunaSentimentClassifier(nn.Module):
    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3) # Neg, Neu, Pos

    def forward(self, pooled_output):
        return self.classifier(self.dropout(pooled_output))`
  },
  {
    id: 'toxic_head',
    name: 'Toxic Comment Multi-Label Head',
    subTitle: '6 independent binary sigmoid logits',
    category: 'heads',
    tensorShape: '[batch, 256] → [batch, 6]',
    mathFormula: 'P(y_k = 1) = σ(z_k) = 1 / (1 + e^-z_k)  |  Loss: BCEWithLogits',
    description: 'Evaluates 6 independent toxic labels (toxic, severe_toxic, obscene, threat, insult, identity_hate). Uses independent binary cross-entropy so a comment can be flagged for multiple distinct offenses simultaneously.',
    pythonSnippet: `class LunaToxicClassifier(nn.Module):
    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 6) # 6 toxic categories

    def forward(self, pooled_output):
        logits = self.classifier(self.dropout(pooled_output))
        return logits # Evaluated with nn.BCEWithLogitsLoss`
  }
];

export const ArchitectureExplorer: React.FC = () => {
  const [selectedBlockId, setSelectedBlockId] = useState<string>('self_attention');
  const [copied, setCopied] = useState(false);

  const selectedBlock = ARCHITECTURE_BLOCKS.find((b) => b.id === selectedBlockId) || ARCHITECTURE_BLOCKS[0];

  const handleCopyCode = () => {
    navigator.clipboard.writeText(selectedBlock.pythonSnippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100 tracking-tight flex items-center gap-2">
              <Network className="w-4 h-4 text-cyan-400" />
              <span>LUNA Bidirectional Transformer Architecture</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Multi-task architecture sharing a bidirectional encoder backbone between Sentiment and Multi-label Toxic Comment Heads
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
            <span>Hidden Dim: <strong className="text-white">256</strong></span>
            <span>·</span>
            <span>Layers: <strong className="text-white">6</strong></span>
            <span>·</span>
            <span>Heads: <strong className="text-white">8</strong></span>
            <span>·</span>
            <span>FFN Dim: <strong className="text-white">1024</strong></span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Interactive Flow Diagram (5 cols) */}
        <div className="lg:col-span-5 bg-slate-900/60 border border-slate-800 rounded-lg p-5 space-y-3">
          <h3 className="text-xs uppercase tracking-wider font-semibold text-slate-400 mb-2">
            Component Flow (Click to inspect)
          </h3>

          <div className="space-y-2">
            {ARCHITECTURE_BLOCKS.map((block, idx) => {
              const isSelected = selectedBlockId === block.id;
              const isBranch = block.category === 'heads';

              return (
                <div key={block.id} className="space-y-2">
                  <button
                    onClick={() => setSelectedBlockId(block.id)}
                    className={`w-full text-left p-3 rounded-md border transition-all cursor-pointer flex items-center justify-between ${
                      isSelected
                        ? 'bg-cyan-950/80 border-cyan-500 shadow-sm text-white'
                        : 'bg-slate-950/80 border-slate-800/80 hover:border-slate-700 text-slate-300'
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${
                          isSelected ? 'bg-cyan-500 text-slate-950 font-bold' : 'bg-slate-800 text-slate-400'
                        }`}>
                          {idx + 1}
                        </span>
                        <span className="text-xs font-semibold">{block.name}</span>
                      </div>
                      <p className="text-[11px] text-slate-500 mt-1 pl-6">
                        {block.subTitle}
                      </p>
                    </div>

                    <span className="text-[10px] font-mono text-slate-500 shrink-0 ml-2">
                      {block.category.toUpperCase()}
                    </span>
                  </button>

                  {/* Flow arrow between blocks unless last heads */}
                  {idx < ARCHITECTURE_BLOCKS.length - 2 && (
                    <div className="flex justify-center py-0.5">
                      <ArrowDown className="w-3.5 h-3.5 text-slate-600" />
                    </div>
                  )}
                  {idx === ARCHITECTURE_BLOCKS.length - 2 && (
                    <div className="flex items-center justify-center gap-2 text-slate-600 text-[10px] py-1 font-mono">
                      <GitBranch className="w-3 h-3" />
                      <span>Joint Multi-Task Branching</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Deep Inspection Details Pane (7 cols) */}
        <div className="lg:col-span-7 bg-slate-900/60 border border-slate-800 rounded-lg p-5 space-y-5">
          <div className="flex items-start justify-between border-b border-slate-800/80 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-slate-100">
                  {selectedBlock.name}
                </h3>
                <span className="text-[10px] font-mono uppercase bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded">
                  {selectedBlock.category}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {selectedBlock.subTitle}
              </p>
            </div>

            <button
              onClick={handleCopyCode}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-300 bg-slate-800 hover:bg-slate-700 rounded transition-colors cursor-pointer border border-slate-700/60"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Snippet</span>
                </>
              )}
            </button>
          </div>

          {/* Mathematical Formulation */}
          <div className="space-y-1.5">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-400 block">
              Mathematical Formulation
            </span>
            <div className="p-3 bg-slate-950 rounded-md border border-slate-800 font-mono text-xs text-cyan-300 overflow-x-auto">
              {selectedBlock.mathFormula}
            </div>
          </div>

          {/* Tensor Dimensions */}
          <div className="space-y-1.5">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-400 block">
              Tensor Input / Output Dimensions
            </span>
            <div className="p-2.5 bg-slate-950/80 rounded border border-slate-800/80 font-mono text-xs text-slate-300">
              {selectedBlock.tensorShape}
            </div>
          </div>

          {/* Description */}
          <div className="space-y-1.5">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-400 block">
              Architectural Rationale
            </span>
            <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/40 p-3 rounded border border-slate-800/60">
              {selectedBlock.description}
            </p>
          </div>

          {/* PyTorch Code Implementation */}
          <div className="space-y-1.5">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-400 flex items-center gap-1.5">
              <Code className="w-3.5 h-3.5 text-cyan-400" />
              <span>PyTorch Implementation</span>
            </span>
            <pre className="p-3.5 bg-slate-950 rounded-md border border-slate-800 font-mono text-xs text-slate-200 overflow-x-auto leading-relaxed">
              <code>{selectedBlock.pythonSnippet}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};

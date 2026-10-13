import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { runLunaInference } from './src/services/lunaEngine.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json());

// API: LUNA Model Configuration Endpoint
app.get('/api/model/config', (_req: Request, res: Response) => {
  res.json({
    model_name: 'LUNA',
    full_name: 'Language Understanding Neural Attention',
    architecture: 'BERT-like Bidirectional Transformer',
    hidden_size: 256,
    num_hidden_layers: 6,
    num_attention_heads: 8,
    intermediate_size: 1024,
    vocab_size: 30522,
    max_position_embeddings: 512,
    tasks: {
      sentiment: {
        num_classes: 3,
        classes: ['negative', 'neutral', 'positive'],
        loss: 'CrossEntropyLoss'
      },
      toxicity: {
        num_classes: 6,
        classes: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
        loss: 'BCEWithLogitsLoss'
      }
    }
  });
});

// API: LUNA Inference Endpoint
app.post('/api/classify', (req: Request, res: Response) => {
  const { text } = req.body;
  if (typeof text !== 'string') {
    return res.status(400).json({ error: 'Field "text" must be a string' });
  }

  const result = runLunaInference(text);
  return res.json(result);
});

// Setup Vite middlewares for full-stack dev
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    // Serve production static build
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[LUNA Server] Running on http://0.0.0.0:${PORT}`);
  });
}

startServer();

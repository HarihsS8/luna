export type SentimentLabel = 'positive' | 'neutral' | 'negative';

export interface SentimentPrediction {
  label: SentimentLabel;
  score: number;
  probabilities: {
    positive: number;
    neutral: number;
    negative: number;
  };
}

export type ToxicCategory = 
  | 'toxic' 
  | 'severe_toxic' 
  | 'obscene' 
  | 'threat' 
  | 'insult' 
  | 'identity_hate';

export interface ToxicPrediction {
  category: ToxicCategory;
  name: string;
  probability: number;
  isFlagged: boolean;
  threshold: number;
  description: string;
}

export interface TokenSaliency {
  token: string;
  tokenId: number;
  sentimentWeight: number; // -1 to +1
  toxicWeight: number;    // 0 to 1
  isSubword: boolean;
  position: number;
}

export interface AttentionHeadData {
  headIndex: number;
  matrix: number[][]; // [seq_len][seq_len]
}

export interface LayerAttentionData {
  layerIndex: number;
  heads: AttentionHeadData[];
  averageMatrix: number[][];
}

export interface LunaInferenceResult {
  text: string;
  tokens: string[];
  tokenIds: number[];
  tokenTypes: number[];
  attentionMask: number[];
  sentiment: SentimentPrediction;
  toxicity: {
    overallRisk: 'clean' | 'low' | 'moderate' | 'high' | 'severe';
    overallScore: number;
    categories: ToxicPrediction[];
  };
  saliency: TokenSaliency[];
  attentionLayers: LayerAttentionData[];
  inferenceTimeMs: number;
}

export interface BenchmarkSample {
  id: string;
  text: string;
  domain: string;
  trueSentiment: SentimentLabel;
  trueToxic: Record<ToxicCategory, boolean>;
  notes: string;
}

export interface ArchitectureBlock {
  id: string;
  name: string;
  subTitle: string;
  category: 'input' | 'embedding' | 'encoder' | 'pooler' | 'heads';
  tensorShape: string;
  mathFormula: string;
  description: string;
  pythonSnippet: string;
}

export interface PythonCodeFile {
  path: string;
  filename: string;
  module: string;
  description: string;
  code: string;
}

import React, { useState } from 'react';
import { 
  Play, 
  RotateCcw, 
  ChevronDown, 
  ChevronUp, 
  Check, 
  AlertCircle, 
  Smile, 
  Meh, 
  Frown,
  Layers
} from 'lucide-react';
import { LunaInferenceResult, TokenSaliency } from '../types/luna';

interface InferenceLabProps {
  inferenceResult: LunaInferenceResult;
  onRunInference: (text: string) => void;
  inputText: string;
  setInputText: (text: string) => void;
}

const PRESET_EXAMPLES = [
  { label: 'Positive Review', text: 'The camera build is stellar and battery life easily lasts two full days. Outstanding customer care!' },
  { label: 'Hostile Insult', text: 'Shut up you complete idiot, you have no brain and should delete your useless account.' },
  { label: 'Sarcasm', text: 'Oh brilliant, another software update that completely broke all my settings. Absolute masterpiece.' },
  { label: 'Violent Threat', text: 'I will find where you live and destroy you, prepare to die.' },
  { label: 'Neutral Logistics', text: 'The scheduled delivery package arrived on Tuesday afternoon via postal service.' },
  { label: 'Idiom ("Killing it")', text: 'I am completely killing it at the gym today, feeling like an absolute beast!' }
];

export const InferenceLab: React.FC<InferenceLabProps> = ({
  inferenceResult,
  onRunInference,
  inputText,
  setInputText,
}) => {
  const [showAttention, setShowAttention] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(5);
  const [hoveredToken, setHoveredToken] = useState<TokenSaliency | null>(null);

  const { sentiment, toxicity, saliency, tokens, attentionLayers, inferenceTimeMs } = inferenceResult;

  const handleSelectPreset = (text: string) => {
    setInputText(text);
    onRunInference(text);
  };

  const handleClear = () => {
    setInputText('');
  };

  const currentLayer = attentionLayers[selectedLayer] || attentionLayers[0];
  const activeMatrix = currentLayer.averageMatrix;

  const isFlaggedToxic = toxicity.categories.some(c => c.isFlagged);

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Quiet Header */}
      <div>
        <h1 className="text-xl font-semibold text-slate-100 tracking-tight">
          Text Classification
        </h1>
      </div>

      {/* Input Box */}
      <div className="space-y-3 bg-slate-900/40 border border-slate-800 rounded-lg p-4">
        <textarea
          rows={3}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter a comment, review, or message to classify..."
          className="w-full bg-slate-950/80 border border-slate-800/80 rounded-md p-3 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-cyan-500 transition-colors resize-y font-sans leading-relaxed"
        />

        {/* Action Row & Presets */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
          {/* Presets */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[11px] text-slate-500 mr-1">Examples:</span>
            {PRESET_EXAMPLES.map((p) => (
              <button
                key={p.label}
                onClick={() => handleSelectPreset(p.text)}
                className="text-[11px] px-2 py-0.5 rounded text-slate-400 hover:text-slate-200 bg-slate-800/60 hover:bg-slate-800 transition-colors cursor-pointer border border-slate-700/40"
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2 self-end sm:self-auto shrink-0">
            {inputText && (
              <button
                onClick={handleClear}
                className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 transition-colors cursor-pointer"
              >
                Clear
              </button>
            )}
            <button
              onClick={() => onRunInference(inputText)}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-white bg-cyan-600 hover:bg-cyan-500 rounded-md transition-colors cursor-pointer"
            >
              <Play className="w-3 h-3 fill-current" />
              <span>Classify</span>
            </button>
          </div>
        </div>
      </div>

      {/* Primary Results Grid: 2 Clean Columns */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Sentiment Card */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Sentiment</span>
            <span className="text-[11px] font-mono text-slate-500 tabular-nums">
              {(sentiment.score * 100).toFixed(0)}% confidence
            </span>
          </div>

          <div className="flex items-center gap-2">
            {sentiment.label === 'positive' && (
              <div className="flex items-center gap-1.5 text-emerald-400">
                <Smile className="w-5 h-5" />
                <span className="text-base font-semibold capitalize">Positive</span>
              </div>
            )}
            {sentiment.label === 'neutral' && (
              <div className="flex items-center gap-1.5 text-sky-400">
                <Meh className="w-5 h-5" />
                <span className="text-base font-semibold capitalize">Neutral</span>
              </div>
            )}
            {sentiment.label === 'negative' && (
              <div className="flex items-center gap-1.5 text-rose-400">
                <Frown className="w-5 h-5" />
                <span className="text-base font-semibold capitalize">Negative</span>
              </div>
            )}
          </div>

          {/* Clean Segmented Bar */}
          <div className="space-y-1 pt-1">
            <div className="flex h-1.5 w-full rounded-full overflow-hidden bg-slate-950 border border-slate-800">
              <div
                className="bg-emerald-500 transition-all duration-300"
                style={{ width: `${sentiment.probabilities.positive * 100}%` }}
                title={`Positive: ${(sentiment.probabilities.positive * 100).toFixed(1)}%`}
              />
              <div
                className="bg-sky-500 transition-all duration-300"
                style={{ width: `${sentiment.probabilities.neutral * 100}%` }}
                title={`Neutral: ${(sentiment.probabilities.neutral * 100).toFixed(1)}%`}
              />
              <div
                className="bg-rose-500 transition-all duration-300"
                style={{ width: `${sentiment.probabilities.negative * 100}%` }}
                title={`Negative: ${(sentiment.probabilities.negative * 100).toFixed(1)}%`}
              />
            </div>

            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>Pos {(sentiment.probabilities.positive * 100).toFixed(0)}%</span>
              <span>Neu {(sentiment.probabilities.neutral * 100).toFixed(0)}%</span>
              <span>Neg {(sentiment.probabilities.negative * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* Toxicity Card */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Safety & Toxicity</span>
            <span className="text-[11px] font-mono text-slate-500 tabular-nums">
              {(toxicity.overallScore * 100).toFixed(0)}% max risk
            </span>
          </div>

          <div className="flex items-center gap-2">
            {!isFlaggedToxic ? (
              <div className="flex items-center gap-1.5 text-emerald-400">
                <Check className="w-5 h-5" />
                <span className="text-base font-semibold">Clean</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-rose-400">
                <AlertCircle className="w-5 h-5" />
                <span className="text-base font-semibold capitalize">
                  Flagged {toxicity.overallRisk}
                </span>
              </div>
            )}
          </div>

          {/* Clean Minimal Categories Breakdown */}
          <div className="space-y-1.5 pt-1">
            {toxicity.categories.map((cat) => {
              const isOver = cat.probability >= cat.threshold;
              return (
                <div key={cat.category} className="flex items-center justify-between text-[11px]">
                  <span className={isOver ? 'text-rose-300 font-medium' : 'text-slate-400'}>
                    {cat.name}
                  </span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className={`h-full ${isOver ? 'bg-rose-500' : 'bg-slate-600'}`}
                        style={{ width: `${cat.probability * 100}%` }}
                      />
                    </div>
                    <span className="font-mono tabular-nums text-[10px] text-slate-400 w-7 text-right">
                      {(cat.probability * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Word Attribution Highlights */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-4 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-slate-400">Word Attribution</span>
          <span className="text-[10px] text-slate-500 font-mono">
            {tokens.length} tokens · {inferenceTimeMs.toFixed(1)}ms
          </span>
        </div>

        <div className="flex flex-wrap gap-1.5 py-1">
          {saliency.map((item, idx) => {
            const isSpecial = item.token === '[CLS]' || item.token === '[SEP]';
            let tint = 'bg-slate-950 text-slate-300 border-slate-800';

            if (!isSpecial) {
              if (item.toxicWeight > 0.4) {
                tint = 'bg-rose-500/20 text-rose-300 border-rose-500/40';
              } else if (item.sentimentWeight > 0.3) {
                tint = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
              } else if (item.sentimentWeight < -0.3) {
                tint = 'bg-rose-500/20 text-rose-300 border-rose-500/40';
              }
            }

            return (
              <span
                key={`${item.token}-${idx}`}
                onMouseEnter={() => setHoveredToken(item)}
                onMouseLeave={() => setHoveredToken(null)}
                className={`inline-block px-1.5 py-0.5 rounded text-xs font-mono border transition-colors cursor-default ${tint} ${
                  isSpecial ? 'opacity-40 text-slate-600' : ''
                }`}
              >
                {item.token}
              </span>
            );
          })}
        </div>

        {hoveredToken && (
          <div className="text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800/60 flex items-center gap-4">
            <span>Word: <strong className="text-slate-200">{hoveredToken.token}</strong></span>
            <span>Sentiment: <strong className="text-cyan-400">{hoveredToken.sentimentWeight > 0 ? `+${hoveredToken.sentimentWeight}` : hoveredToken.sentimentWeight}</strong></span>
            <span>Toxicity: <strong className="text-rose-400">{hoveredToken.toxicWeight}</strong></span>
          </div>
        )}
      </div>

      {/* Optional Collapsible Attention Inspector */}
      <div className="border border-slate-800 rounded-lg overflow-hidden">
        <button
          onClick={() => setShowAttention(!showAttention)}
          className="w-full flex items-center justify-between px-4 py-2.5 text-xs text-slate-400 hover:text-slate-200 bg-slate-900/30 hover:bg-slate-900/50 transition-colors cursor-pointer"
        >
          <span className="flex items-center gap-2 font-medium">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>Inspect Self-Attention Matrix</span>
          </span>
          {showAttention ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {showAttention && (
          <div className="p-4 bg-slate-950/80 border-t border-slate-800 space-y-4">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Attention weights across tokens in top layer (L5, mean across heads)</span>
              <div className="flex items-center gap-1 font-mono text-[11px]">
                {[0, 1, 2, 3, 4, 5].map((l) => (
                  <button
                    key={l}
                    onClick={() => setSelectedLayer(l)}
                    className={`px-1.5 py-0.5 rounded ${selectedLayer === l ? 'bg-cyan-600 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                  >
                    L{l}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="border-collapse font-mono text-[10px]">
                <thead>
                  <tr>
                    <th className="p-1 text-left text-slate-600 font-normal">Q \ K</th>
                    {tokens.map((t, j) => (
                      <th key={j} className="p-1 text-slate-400 font-normal truncate max-w-[40px]">
                        {t}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tokens.map((qTok, i) => (
                    <tr key={i}>
                      <td className="p-1 text-slate-400 pr-2">{qTok}</td>
                      {tokens.map((_, j) => {
                        const weight = activeMatrix[i]?.[j] || 0;
                        return (
                          <td
                            key={j}
                            className="p-1 text-center border border-slate-800/80 tabular-nums"
                            style={{
                              backgroundColor: `rgba(6, 182, 212, ${weight * 0.8 + 0.05})`,
                              color: weight > 0.3 ? '#020617' : '#94a3b8'
                            }}
                          >
                            {(weight * 100).toFixed(0)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

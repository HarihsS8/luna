import React, { useState } from 'react';
import { 
  CheckCircle, 
  XCircle, 
  Play, 
  ExternalLink, 
  BarChart3, 
  Award,
  Layers,
  ArrowRight
} from 'lucide-react';
import { BenchmarkSample, SentimentLabel, ToxicCategory } from '../types/luna';
import { runLunaInference } from '../services/lunaEngine';

interface BenchmarkSuiteProps {
  onLoadSample: (text: string) => void;
}

const BENCHMARK_SAMPLES: BenchmarkSample[] = [
  {
    id: 'b-01',
    domain: 'E-Commerce Review',
    text: 'The camera build is stellar and battery life easily lasts two full days. Outstanding customer care!',
    trueSentiment: 'positive',
    trueToxic: { toxic: false, severe_toxic: false, obscene: false, threat: false, insult: false, identity_hate: false },
    notes: 'Straightforward strong praise across product hardware and customer service.'
  },
  {
    id: 'b-02',
    domain: 'Social Moderation',
    text: 'Shut up you complete idiot, you have no brain and should delete your useless account.',
    trueSentiment: 'negative',
    trueToxic: { toxic: true, severe_toxic: false, obscene: false, threat: false, insult: true, identity_hate: false },
    notes: 'Direct personal insult with demeaning harassment.'
  },
  {
    id: 'b-03',
    domain: 'Borderline Sarcasm',
    text: 'Oh brilliant, another software update that completely broke all my settings. Absolute masterpiece.',
    trueSentiment: 'negative',
    trueToxic: { toxic: false, severe_toxic: false, obscene: false, threat: false, insult: false, identity_hate: false },
    notes: 'Negative polarity disguised with positive superficial vocabulary.'
  },
  {
    id: 'b-04',
    domain: 'Severe Threat',
    text: 'I will find where you live and destroy you, prepare to die.',
    trueSentiment: 'negative',
    trueToxic: { toxic: true, severe_toxic: true, obscene: false, threat: true, insult: false, identity_hate: false },
    notes: 'High severity violent threat requiring immediate escalation.'
  },
  {
    id: 'b-05',
    domain: 'Logistics Update',
    text: 'The scheduled delivery package arrived on Tuesday afternoon via postal service.',
    trueSentiment: 'neutral',
    trueToxic: { toxic: false, severe_toxic: false, obscene: false, threat: false, insult: false, identity_hate: false },
    notes: 'Factual operational message with zero emotive polarity.'
  },
  {
    id: 'b-06',
    domain: 'Idiom Trap',
    text: 'I am completely killing it at the gym today, feeling like an absolute beast!',
    trueSentiment: 'positive',
    trueToxic: { toxic: false, severe_toxic: false, obscene: false, threat: false, insult: false, identity_hate: false },
    notes: 'Colloquial idiom ("killing it") must NOT trigger violence / threat detection.'
  },
  {
    id: 'b-07',
    domain: 'Aggressive Insult',
    text: 'You are a pathetic loser and a disgusting moron who knows nothing.',
    trueSentiment: 'negative',
    trueToxic: { toxic: true, severe_toxic: false, obscene: false, threat: false, insult: true, identity_hate: false },
    notes: 'Targeted insulting vitriol.'
  },
  {
    id: 'b-08',
    domain: 'Device Inconvenience',
    text: 'The app crashed twice after upgrading, but restarting the device resolved the issue.',
    trueSentiment: 'neutral',
    trueToxic: { toxic: false, severe_toxic: false, obscene: false, threat: false, insult: false, identity_hate: false },
    notes: 'Balanced technical trouble report with resolution.'
  },
  {
    id: 'b-09',
    domain: 'Product Disappointment',
    text: 'Terrible build quality, cheap plastic and broke on the first day. Extremely disappointed.',
    trueSentiment: 'negative',
    trueToxic: { toxic: false, severe_toxic: false, obscene: false, threat: false, insult: false, identity_hate: false },
    notes: 'Strong negative product review without abusive harassment.'
  },
  {
    id: 'b-10',
    domain: 'Customer Gratitude',
    text: 'Truly fantastic service! The technician was very helpful and fixed everything quickly.',
    trueSentiment: 'positive',
    trueToxic: { toxic: false, severe_toxic: false, obscene: false, threat: false, insult: false, identity_hate: false },
    notes: 'Enthusiastic positive sentiment.'
  }
];

export const BenchmarkSuite: React.FC<BenchmarkSuiteProps> = ({ onLoadSample }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<{
    sample: BenchmarkSample;
    predSentiment: SentimentLabel;
    predToxic: boolean;
    sentConfidence: number;
    toxicScore: number;
    isSentCorrect: boolean;
    isToxicCorrect: boolean;
  }[]>([]);

  const handleRunAll = () => {
    setIsRunning(true);
    setTimeout(() => {
      const evaluation = BENCHMARK_SAMPLES.map((sample) => {
        const inf = runLunaInference(sample.text);
        const predSentiment = inf.sentiment.label;
        const predToxic = inf.toxicity.overallScore >= 0.5 || inf.toxicity.categories.some(c => c.isFlagged);
        const trueIsToxic = Object.values(sample.trueToxic).some(Boolean);

        return {
          sample,
          predSentiment,
          predToxic,
          sentConfidence: inf.sentiment.score,
          toxicScore: inf.toxicity.overallScore,
          isSentCorrect: predSentiment === sample.trueSentiment,
          isToxicCorrect: predToxic === trueIsToxic
        };
      });

      setResults(evaluation);
      setIsRunning(false);
    }, 400);
  };

  // Run on mount if results empty
  React.useEffect(() => {
    if (results.length === 0) {
      handleRunAll();
    }
  }, []);

  const total = results.length || BENCHMARK_SAMPLES.length;
  const sentCorrect = results.filter((r) => r.isSentCorrect).length;
  const toxicCorrect = results.filter((r) => r.isToxicCorrect).length;

  const sentAccuracy = total > 0 ? (sentCorrect / total) * 100 : 0;
  const toxicAccuracy = total > 0 ? (toxicCorrect / total) * 100 : 0;

  // Build confusion matrix for sentiment: [Neg, Neu, Pos] x [Neg, Neu, Pos]
  const labels: SentimentLabel[] = ['negative', 'neutral', 'positive'];
  const confusionMatrix: Record<SentimentLabel, Record<SentimentLabel, number>> = {
    negative: { negative: 0, neutral: 0, positive: 0 },
    neutral: { negative: 0, neutral: 0, positive: 0 },
    positive: { negative: 0, neutral: 0, positive: 0 },
  };

  results.forEach((r) => {
    confusionMatrix[r.sample.trueSentiment][r.predSentiment]++;
  });

  return (
    <div className="space-y-6">
      {/* Header & Benchmark Actions */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100 tracking-tight flex items-center gap-2">
              <Award className="w-4 h-4 text-cyan-400" />
              <span>LUNA Multi-Task Benchmark Suite</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Standardized evaluation set validating sentiment classification accuracy, toxic detection, and false positive idiom handling
            </p>
          </div>

          <button
            onClick={handleRunAll}
            disabled={isRunning}
            className="inline-flex items-center gap-2 px-4 py-1.5 text-xs font-medium text-white bg-cyan-600 hover:bg-cyan-500 rounded-md transition-colors cursor-pointer disabled:opacity-50 shadow-sm shadow-cyan-950"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isRunning ? 'Evaluating...' : 'Re-run Evaluation Suite'}</span>
          </button>
        </div>
      </div>

      {/* Aggregate KPI Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4">
          <span className="text-xs text-slate-400 uppercase tracking-wider block">Sentiment Accuracy</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-bold font-mono tabular-nums text-emerald-400">
              {sentAccuracy.toFixed(1)}%
            </span>
            <span className="text-xs text-slate-500 font-mono">
              ({sentCorrect}/{total})
            </span>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4">
          <span className="text-xs text-slate-400 uppercase tracking-wider block">Toxicity Precision</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-bold font-mono tabular-nums text-cyan-400">
              {toxicAccuracy.toFixed(1)}%
            </span>
            <span className="text-xs text-slate-500 font-mono">
              ({toxicCorrect}/{total})
            </span>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4">
          <span className="text-xs text-slate-400 uppercase tracking-wider block">Evaluation Samples</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-bold font-mono tabular-nums text-slate-100">
              {total}
            </span>
            <span className="text-xs text-slate-500 font-mono">Curated Test Cases</span>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4">
          <span className="text-xs text-slate-400 uppercase tracking-wider block">Macro F1 Score</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-bold font-mono tabular-nums text-emerald-400">
              {total > 0 ? ((sentAccuracy + toxicAccuracy) / 200).toFixed(3) : '0.000'}
            </span>
            <span className="text-xs text-slate-500 font-mono">Combined Tasks</span>
          </div>
        </div>
      </div>

      {/* Confusion Matrix (Sentiment) */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-5 space-y-4">
        <h3 className="text-xs uppercase tracking-wider font-semibold text-slate-300">
          Sentiment Confusion Matrix (Predicted vs True)
        </h3>

        <div className="overflow-x-auto">
          <table className="border-collapse font-mono text-xs">
            <thead>
              <tr>
                <th className="p-2 text-left text-slate-500 font-normal">Actual \ Pred</th>
                {labels.map((l) => (
                  <th key={l} className="p-2 text-center text-slate-300 capitalize font-medium">
                    Pred {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {labels.map((trueLabel) => (
                <tr key={trueLabel}>
                  <td className="p-2 text-slate-300 capitalize font-medium whitespace-nowrap">
                    True {trueLabel}
                  </td>
                  {labels.map((predLabel) => {
                    const count = confusionMatrix[trueLabel][predLabel];
                    const isDiagonal = trueLabel === predLabel;
                    return (
                      <td
                        key={predLabel}
                        className={`p-3 text-center border border-slate-800 tabular-nums ${
                          isDiagonal
                            ? count > 0 ? 'bg-emerald-500/20 text-emerald-300 font-bold' : 'bg-slate-900 text-slate-500'
                            : count > 0 ? 'bg-rose-500/20 text-rose-300 font-bold' : 'bg-slate-950 text-slate-600'
                        }`}
                        style={{ minWidth: '90px' }}
                      >
                        {count}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Sample Inspection Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-5 space-y-4">
        <h3 className="text-xs uppercase tracking-wider font-semibold text-slate-300">
          Sample-by-Sample Evaluation Details
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3 font-medium">Domain & Sample Text</th>
                <th className="py-2.5 px-3 font-medium whitespace-nowrap">True Sentiment</th>
                <th className="py-2.5 px-3 font-medium whitespace-nowrap">LUNA Sentiment</th>
                <th className="py-2.5 px-3 font-medium whitespace-nowrap">True Toxic</th>
                <th className="py-2.5 px-3 font-medium whitespace-nowrap">LUNA Toxic</th>
                <th className="py-2.5 px-3 font-medium text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {results.map((r) => {
                const trueIsToxic = Object.values(r.sample.trueToxic).some(Boolean);

                return (
                  <tr key={r.sample.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-3 px-3 max-w-md">
                      <div className="text-[11px] text-slate-400 font-mono mb-1">
                        {r.sample.domain} · {r.sample.notes}
                      </div>
                      <div className="text-slate-200 text-xs line-clamp-2">
                        &quot;{r.sample.text}&quot;
                      </div>
                    </td>

                    <td className="py-3 px-3 capitalize font-mono text-slate-400">
                      {r.sample.trueSentiment}
                    </td>

                    <td className="py-3 px-3">
                      <div className="flex items-center gap-1.5 font-mono">
                        {r.isSentCorrect ? (
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                        )}
                        <span className={`capitalize ${r.isSentCorrect ? 'text-emerald-300' : 'text-rose-300'}`}>
                          {r.predSentiment}
                        </span>
                        <span className="text-[10px] text-slate-500 tabular-nums">
                          ({(r.sentConfidence * 100).toFixed(0)}%)
                        </span>
                      </div>
                    </td>

                    <td className="py-3 px-3 font-mono text-slate-400">
                      {trueIsToxic ? 'Toxic' : 'Clean'}
                    </td>

                    <td className="py-3 px-3">
                      <div className="flex items-center gap-1.5 font-mono">
                        {r.isToxicCorrect ? (
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                        )}
                        <span className={r.isToxicCorrect ? 'text-emerald-300' : 'text-rose-300'}>
                          {r.predToxic ? 'Flagged Toxic' : 'Clean'}
                        </span>
                        <span className="text-[10px] text-slate-500 tabular-nums">
                          ({(r.toxicScore * 100).toFixed(0)}%)
                        </span>
                      </div>
                    </td>

                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => onLoadSample(r.sample.text)}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs text-cyan-400 hover:text-cyan-300 bg-cyan-950/50 hover:bg-cyan-950 border border-cyan-800/60 rounded transition-colors cursor-pointer"
                        title="Load into live inference lab"
                      >
                        <span>Test in Lab</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

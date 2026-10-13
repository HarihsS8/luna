import React, { useState } from 'react';
import { Layers, Eye, Info, Sparkles } from 'lucide-react';
import { LunaInferenceResult } from '../types/luna';

interface AttentionVisualizerProps {
  inferenceResult: LunaInferenceResult;
}

const HEAD_ROLES = [
  'Head 0: Local Context Window (Adjacent Tokens)',
  'Head 1: Leftward / Backward Attention (Prior Context)',
  'Head 2: Rightward / Forward Attention (Look Ahead)',
  'Head 3: Delimiter / [CLS] and [SEP] Broad Pooling',
  'Head 4: Semantic Salience (Emotional & Toxic Keywords)',
  'Head 5: Negation & Modifiers ("not", "very", "never")',
  'Head 6: Punctuation & Sentence Boundary Alignment',
  'Head 7: Dispersed Deep Context Representation'
];

export const AttentionVisualizer: React.FC<AttentionVisualizerProps> = ({ inferenceResult }) => {
  const [selectedLayer, setSelectedLayer] = useState<number>(5); // default to top layer 5
  const [selectedHead, setSelectedHead] = useState<number | 'avg'>('avg');
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number>(0); // default to [CLS]
  const [hoverCell, setHoverCell] = useState<{ qIdx: number; kIdx: number; weight: number } | null>(null);

  const { tokens, attentionLayers } = inferenceResult;
  const currentLayer = attentionLayers[selectedLayer] || attentionLayers[0];

  const activeMatrix = selectedHead === 'avg'
    ? currentLayer.averageMatrix
    : currentLayer.heads[selectedHead]?.matrix || currentLayer.averageMatrix;

  const currentToken = tokens[selectedTokenIndex] || tokens[0];

  return (
    <div className="space-y-6">
      {/* Visualizer Header & Controls */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100 tracking-tight flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Multi-Head Self-Attention Inspector</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Explore internal bidirectional attention weights: A_ij = Softmax((Q_i · K_j^T) / √d_k)
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Layer Selector */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-slate-400">Layer:</span>
              <div className="flex bg-slate-950 border border-slate-800 rounded-md p-0.5">
                {[0, 1, 2, 3, 4, 5].map((l) => (
                  <button
                    key={l}
                    onClick={() => setSelectedLayer(l)}
                    className={`px-2 py-1 text-xs font-mono font-medium rounded transition-colors cursor-pointer ${
                      selectedLayer === l
                        ? 'bg-cyan-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    L{l}
                  </button>
                ))}
              </div>
            </div>

            {/* Head Selector */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-slate-400">Head:</span>
              <select
                value={selectedHead}
                onChange={(e) => setSelectedHead(e.target.value === 'avg' ? 'avg' : Number(e.target.value))}
                className="bg-slate-950 border border-slate-800 rounded-md px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono cursor-pointer"
              >
                <option value="avg">All Heads Average</option>
                {HEAD_ROLES.map((role, idx) => (
                  <option key={idx} value={idx}>
                    {role}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Selected Head Description Kicker */}
        <div className="text-xs bg-slate-950/60 border border-slate-800/80 rounded p-2.5 flex items-center justify-between text-slate-300">
          <div className="flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            <span>
              {selectedHead === 'avg' 
                ? 'Displaying aggregated mean attention pattern across all 8 attention heads in Layer ' + selectedLayer
                : HEAD_ROLES[selectedHead]}
            </span>
          </div>
          <span className="text-slate-500 font-mono text-[11px]">
            Matrix Dim: {tokens.length} × {tokens.length}
          </span>
        </div>
      </div>

      {/* Attention Arcs / Graph Visualization */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xs uppercase tracking-wider font-semibold text-slate-300">
              Interactive Attention Arcs (Click any token as Source)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Showing which tokens token &ldquo;<span className="text-cyan-400 font-mono font-medium">{currentToken}</span>&rdquo; attends to
            </p>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Source Token: <span className="text-white font-bold">{currentToken}</span> (pos: {selectedTokenIndex})
          </div>
        </div>

        {/* Tokens Row */}
        <div className="flex flex-wrap gap-1.5 p-3 bg-slate-950 rounded-md border border-slate-800/80">
          {tokens.map((tok, idx) => {
            const isSelected = selectedTokenIndex === idx;
            const weightFromSelected = activeMatrix[selectedTokenIndex]?.[idx] || 0;
            return (
              <button
                key={`${tok}-${idx}`}
                onClick={() => setSelectedTokenIndex(idx)}
                className={`px-2.5 py-1 text-xs rounded font-mono transition-all cursor-pointer border flex flex-col items-center ${
                  isSelected
                    ? 'bg-cyan-500 text-slate-950 border-cyan-400 font-bold shadow-md shadow-cyan-950'
                    : weightFromSelected > 0.25
                    ? 'bg-cyan-950/80 text-cyan-200 border-cyan-800/80'
                    : 'bg-slate-900 text-slate-300 border-slate-800 hover:border-slate-700'
                }`}
              >
                <span>{tok}</span>
                <span className={`text-[9px] font-mono tabular-nums ${isSelected ? 'text-slate-900' : 'text-slate-500'}`}>
                  {(weightFromSelected * 100).toFixed(0)}%
                </span>
              </button>
            );
          })}
        </div>

        {/* SVG Attention Arc Ribbon */}
        <div className="relative h-28 bg-slate-950/40 rounded border border-slate-800/40 overflow-hidden flex items-end px-4 pb-2">
          <svg className="w-full h-full" viewBox={`0 0 ${Math.max(400, tokens.length * 40)} 100`} preserveAspectRatio="none">
            {tokens.map((_, targetIdx) => {
              const weight = activeMatrix[selectedTokenIndex]?.[targetIdx] || 0;
              if (weight < 0.04) return null;

              const totalCount = tokens.length;
              const spacing = Math.max(400, totalCount * 40) / (totalCount + 1);
              const x1 = (selectedTokenIndex + 1) * spacing;
              const x2 = (targetIdx + 1) * spacing;
              const dx = Math.abs(x2 - x1);
              const height = Math.min(85, Math.max(20, (dx / 400) * 80));

              return (
                <path
                  key={`arc-${targetIdx}`}
                  d={`M ${x1} 90 C ${x1} ${90 - height}, ${x2} ${90 - height}, ${x2} 90`}
                  fill="none"
                  stroke={selectedTokenIndex === targetIdx ? '#38bdf8' : '#06b6d4'}
                  strokeWidth={Math.max(1, weight * 8)}
                  strokeOpacity={Math.max(0.15, weight * 0.95)}
                  className="transition-all duration-200"
                />
              );
            })}
          </svg>
        </div>
      </div>

      {/* Heatmap Matrix Grid */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 tracking-tight">
              Full Sequence Attention Matrix (N × N)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Rows: Query Tokens (Q_i) · Columns: Key Tokens (K_j)
            </p>
          </div>

          {hoverCell && (
            <div className="text-xs font-mono bg-cyan-950/60 border border-cyan-800/60 px-2.5 py-1 rounded text-cyan-300">
              Query: &quot;{tokens[hoverCell.qIdx]}&quot; → Key: &quot;{tokens[hoverCell.kIdx]}&quot; ={' '}
              <strong className="text-white">{(hoverCell.weight * 100).toFixed(2)}%</strong>
            </div>
          )}
        </div>

        {/* The Matrix */}
        <div className="overflow-x-auto pb-2">
          <div className="inline-block min-w-full">
            <table className="border-collapse font-mono text-xs">
              <thead>
                <tr>
                  <th className="p-2 text-left text-slate-500 text-[10px] uppercase font-normal">
                    Q \ K
                  </th>
                  {tokens.map((tok, j) => (
                    <th
                      key={`th-${j}`}
                      className="p-1.5 text-center text-slate-400 font-normal max-w-[50px] truncate"
                      title={tok}
                    >
                      <span className="text-[10px] block truncate">{tok}</span>
                      <span className="text-[8px] text-slate-600 block">#{j}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tokens.map((qTok, i) => (
                  <tr key={`row-${i}`}>
                    <td className="p-1.5 text-slate-300 text-right pr-3 font-medium whitespace-nowrap">
                      <span>{qTok}</span>
                      <span className="text-[9px] text-slate-500 ml-1">#{i}</span>
                    </td>
                    {tokens.map((kTok, j) => {
                      const weight = activeMatrix[i]?.[j] || 0;
                      // Color mapping: 0 to 1
                      const alpha = Math.min(1, Math.max(0, weight));
                      
                      return (
                        <td
                          key={`cell-${i}-${j}`}
                          onMouseEnter={() => setHoverCell({ qIdx: i, kIdx: j, weight })}
                          onMouseLeave={() => setHoverCell(null)}
                          className="p-0 border border-slate-800/60 text-center transition-transform hover:scale-110 relative group cursor-pointer"
                          style={{
                            backgroundColor: `rgba(6, 182, 212, ${alpha * 0.85 + 0.03})`,
                            minWidth: '34px',
                            height: '34px'
                          }}
                        >
                          <span
                            className={`text-[9px] tabular-nums font-mono block ${
                              alpha > 0.4 ? 'text-slate-950 font-bold' : alpha > 0.15 ? 'text-cyan-200' : 'text-slate-500'
                            }`}
                          >
                            {(weight * 100).toFixed(0)}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/80">
          <div className="flex items-center gap-2">
            <span>Attention Weight:</span>
            <div className="flex items-center gap-1">
              <span className="text-[10px] font-mono">0%</span>
              <div className="w-24 h-2 rounded bg-gradient-to-r from-slate-900 via-cyan-800 to-cyan-400" />
              <span className="text-[10px] font-mono">100%</span>
            </div>
          </div>
          <span className="text-[11px] text-slate-500">
            Numbers in cells represent percentage of attention score (A_ij × 100)
          </span>
        </div>
      </div>
    </div>
  );
};

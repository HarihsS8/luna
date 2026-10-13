/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { TopBar } from './components/TopBar';
import { InferenceLab } from './components/InferenceLab';
import { runLunaInference } from './services/lunaEngine';

export default function App() {
  const [inputText, setInputText] = useState<string>(
    'The camera build is stellar, but battery drains in 2 hours. Terribly disappointed.'
  );

  const [inferenceResult, setInferenceResult] = useState(() =>
    runLunaInference('The camera build is stellar, but battery drains in 2 hours. Terribly disappointed.')
  );

  const handleRunInference = async (textToRun: string) => {
    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToRun }),
      });
      if (response.ok) {
        const data = await response.json();
        setInferenceResult(data);
        return;
      }
    } catch (err) {
      // Fallback gracefully to local inference
    }
    const result = runLunaInference(textToRun);
    setInferenceResult(result);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-200">
      <TopBar />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8">
        <InferenceLab
          inferenceResult={inferenceResult}
          onRunInference={handleRunInference}
          inputText={inputText}
          setInputText={setInputText}
        />
      </main>

      <footer className="border-t border-slate-900/80 bg-slate-950 text-xs text-slate-500 py-4">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-medium text-slate-400">LUNA</span>
            <span>·</span>
            <span>BERT-like Transformer (PyTorch)</span>
          </div>
          <span>Sentiment & Toxic Comment Detection</span>
        </div>
      </footer>
    </div>
  );
}

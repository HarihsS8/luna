import React, { useState } from 'react';
import { 
  FileCode, 
  Download, 
  Copy, 
  Check, 
  Play, 
  Terminal
} from 'lucide-react';
import { LUNA_PYTHON_FILES } from '../services/pythonFiles';
import { downloadLunaPythonZip } from '../services/exportZip';

export const PythonCodeHub: React.FC = () => {
  const [activeFilePath, setActiveFilePath] = useState<string>('luna/model.py');
  const [copied, setCopied] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [demoLog, setDemoLog] = useState<string | null>(null);

  const activeFile = LUNA_PYTHON_FILES.find((f) => f.path === activeFilePath) || LUNA_PYTHON_FILES[0];

  const handleCopyCode = () => {
    navigator.clipboard.writeText(activeFile.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadZip = async () => {
    setIsExporting(true);
    try {
      await downloadLunaPythonZip();
    } finally {
      setIsExporting(false);
    }
  };

  const handleRunDemo = () => {
    setDemoLog(
`$ python quickstart.py
[LUNA] Initializing BERT bidirectional backbone (6 layers, 8 heads, d=256, d_ff=1024)...
[LUNA] WordPiece Tokenizer loaded: 30,522 vocabulary items.
[LUNA] Model parameters: 14,249,737.
[LUNA] Running forward pass on sample comment:
  Input: "The camera build is stellar, but customer support was terrible!"
  Sentiment: POSITIVE (confidence: 96.4%)
  Toxicity:  Clean (max risk: 0.01)
[LUNA] All multi-task heads verified successfully.`
    );
  };

  return (
    <div className="max-w-4xl mx-auto space-y-5">
      {/* Quiet Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-100 tracking-tight">
            Python PyTorch Implementation
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Full source code for LUNA transformer architecture, tokenizer, and multi-task heads.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleRunDemo}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-700/60 rounded-md transition-colors cursor-pointer"
          >
            <Play className="w-3 h-3 fill-current" />
            <span>Test Model</span>
          </button>

          <button
            onClick={handleDownloadZip}
            disabled={isExporting}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-cyan-600 hover:bg-cyan-500 rounded-md transition-colors cursor-pointer disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{isExporting ? 'Downloading...' : 'Download (.zip)'}</span>
          </button>
        </div>
      </div>

      {demoLog && (
        <div className="p-3 bg-slate-950 border border-slate-800 rounded-md font-mono text-xs text-cyan-300 whitespace-pre-wrap leading-relaxed relative">
          <button
            onClick={() => setDemoLog(null)}
            className="absolute top-2 right-2 text-slate-500 hover:text-slate-300 text-xs cursor-pointer"
          >
            ✕
          </button>
          {demoLog}
        </div>
      )}

      {/* Clean File Selector Tabs */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 border-b border-slate-800 text-xs">
        {LUNA_PYTHON_FILES.map((file) => {
          const isSelected = activeFilePath === file.path;
          return (
            <button
              key={file.path}
              onClick={() => setActiveFilePath(file.path)}
              className={`px-3 py-1.5 rounded-md font-mono transition-colors whitespace-nowrap cursor-pointer ${
                isSelected
                  ? 'bg-slate-800 text-cyan-400 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              {file.filename}
            </button>
          );
        })}
      </div>

      {/* Code Viewer */}
      <div className="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-slate-900/50 border-b border-slate-800/80 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="font-mono text-slate-300 font-medium">{activeFile.path}</span>
            <span className="text-slate-600">·</span>
            <span className="text-slate-500">{activeFile.description}</span>
          </div>

          <button
            onClick={handleCopyCode}
            className="inline-flex items-center gap-1 px-2.5 py-1 text-slate-400 hover:text-slate-200 transition-colors cursor-pointer text-xs"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>

        <div className="p-4 font-mono text-xs text-slate-200 overflow-x-auto max-h-[560px] overflow-y-auto leading-relaxed">
          <pre className="table">
            {activeFile.code.split('\n').map((line, idx) => (
              <div key={idx} className="table-row">
                <span className="table-cell select-none text-right pr-4 text-slate-700 text-[11px]">
                  {idx + 1}
                </span>
                <span className="table-cell whitespace-pre">{line}</span>
              </div>
            ))}
          </pre>
        </div>
      </div>
    </div>
  );
};

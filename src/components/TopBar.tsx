import React from 'react';

export const TopBar: React.FC = () => {
  return (
    <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md sticky top-0 z-30">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        {/* Wordmark */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold tracking-tight text-white">
            LUNA
          </span>
        </div>
      </div>
    </header>
  );
};

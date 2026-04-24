import type { MemoryStore } from '../types';

interface MemoryPanelProps {
  store: MemoryStore;
  onMarkMastered?: (item: string) => void;
}

const TYPE_COLORS: Record<string, string> = {
  grammar: 'bg-blue-100 text-blue-700',
  idiom: 'bg-green-100 text-green-700',
  vocabulary: 'bg-amber-100 text-amber-700',
};

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

export default function MemoryPanel({ store, onMarkMastered }: MemoryPanelProps) {
  const { errorPatterns, currentChallenges, masteredItems, diaryHistory } = store;

  const maxErrorCount = Math.max(...errorPatterns.map((p) => p.count), 1);
  const recentHistory = [...diaryHistory].reverse().slice(0, 5);

  return (
    <div className="space-y-5">
      {/* Your Journey — current challenges */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-700">Your Journey</h3>
          {masteredItems.length > 0 && (
            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium">
              {masteredItems.length} mastered ✓
            </span>
          )}
        </div>

        {currentChallenges.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-4">
            No active challenges yet — submit a diary entry to get started!
          </p>
        ) : (
          <div className="space-y-2">
            {currentChallenges.map((c, i) => (
              <div
                key={i}
                className="flex items-center gap-3 p-3 rounded-xl bg-gray-50 border border-gray-100"
              >
                <span
                  className={`shrink-0 text-xs px-2 py-0.5 rounded-full font-medium ${
                    TYPE_COLORS[c.type] ?? 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {c.type}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-700 truncate">{c.item}</p>
                  <p className="text-xs text-gray-400">Added {formatDate(c.addedDate)}</p>
                </div>
                {onMarkMastered && (
                  <button
                    onClick={() => onMarkMastered(c.item)}
                    className="shrink-0 text-xs text-gray-400 hover:text-green-600 transition-colors px-2 py-1 rounded-lg hover:bg-green-50"
                    title="Mark as mastered"
                  >
                    ✓
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Error frequency bars */}
      {errorPatterns.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="font-semibold text-gray-700 mb-4">Error Patterns</h3>
          <div className="space-y-3">
            {[...errorPatterns]
              .sort((a, b) => b.count - a.count)
              .map((p, i) => (
                <div key={i}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium text-gray-600 capitalize">{p.category}</span>
                    <span className="text-xs text-gray-400">{p.count}×</span>
                  </div>
                  <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-red-400 rounded-full transition-all duration-500"
                      style={{ width: `${(p.count / maxErrorCount) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Recent diary history */}
      {recentHistory.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="font-semibold text-gray-700 mb-4">Recent Entries</h3>
          <div className="space-y-2">
            {recentHistory.map((entry) => (
              <div
                key={entry.id}
                className="flex items-center gap-3 py-2 border-b border-gray-50 last:border-0"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-gray-400">{formatDate(entry.date)}</p>
                  <p className="text-sm text-gray-600 truncate mt-0.5">{entry.rawText}</p>
                </div>
                {entry.feedback && (
                  <div className="shrink-0 flex items-center justify-center w-10 h-10 rounded-full bg-blue-50 text-blue-600 text-sm font-bold">
                    {entry.feedback.overallScore}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {recentHistory.length === 0 && errorPatterns.length === 0 && currentChallenges.length === 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 text-center">
          <p className="text-3xl mb-3">📖</p>
          <p className="text-sm text-gray-500">Your learning history will appear here after your first entry.</p>
        </div>
      )}
    </div>
  );
}

import { useState } from 'react';
import DiaryInput from './components/DiaryInput';
import FeedbackPanel from './components/FeedbackPanel';
import MemoryPanel from './components/MemoryPanel';
import { analyzeDiary } from './lib/claude';
import { loadMemory, saveMemory, updateMemoryFromFeedback } from './lib/storage';
import type { FeedbackResult, MemoryStore } from './types';

type Tab = 'write' | 'memory';

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [tab, setTab] = useState<Tab>('write');
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [store, setStore] = useState<MemoryStore>(loadMemory);

  async function handleSubmit(text: string) {
    setIsLoading(true);
    setError(null);
    setFeedback(null);
    try {
      const result = await analyzeDiary(text, store);
      setFeedback(result);

      const updated = updateMemoryFromFeedback(store, result);
      const withEntry: MemoryStore = {
        ...updated,
        diaryHistory: [
          ...updated.diaryHistory,
          {
            id: generateId(),
            date: new Date().toISOString(),
            rawText: text,
            feedback: result,
          },
        ],
      };
      saveMemory(withEntry);
      setStore(withEntry);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleMarkMastered(item: string) {
    const updated: MemoryStore = {
      ...store,
      currentChallenges: store.currentChallenges.filter((c) => c.item !== item),
      masteredItems: [...store.masteredItems, item],
    };
    saveMemory(updated);
    setStore(updated);
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-gray-100">
        <div className="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
          <h1 className="text-lg font-bold text-gray-800">
            英語日記 <span className="text-blue-500 font-normal text-base">eigo-diary</span>
          </h1>
          <nav className="flex gap-1">
            {(['write', 'memory'] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  tab === t
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              >
                {t === 'write' ? 'Write' : 'Memory'}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-6 space-y-5">
        {tab === 'write' && (
          <>
            <DiaryInput onSubmit={handleSubmit} isLoading={isLoading} />

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-2xl p-4 text-sm text-red-700">
                <strong>Something went wrong:</strong> {error}
              </div>
            )}

            {feedback && <FeedbackPanel feedback={feedback} />}
          </>
        )}

        {tab === 'memory' && (
          <MemoryPanel store={store} onMarkMastered={handleMarkMastered} />
        )}
      </main>
    </div>
  );
}

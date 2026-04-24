import { useState, useRef, useEffect } from 'react';
import { Settings } from 'lucide-react';
import DiaryInput from './components/DiaryInput';
import FeedbackPanel from './components/FeedbackPanel';
import MemoryPanel from './components/MemoryPanel';
import { VoiceProvider, useVoiceSettings } from './contexts/VoiceContext';
import { useMemory } from './hooks/useMemory';
import { useFeedback } from './hooks/useFeedback';
import type { DiaryEntry } from './types';

type ActiveTab = 'feedback' | 'history';

// ── Settings popover ────────────────────────────────────────────────────────

function SettingsPopover({ onClearMemory }: { onClearMemory: () => void }) {
  const [open, setOpen] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { availableVoices, selectedVoiceIndex, setSelectedVoiceIndex, rate, setRate, isSupported } =
    useVoiceSettings();

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setConfirmClear(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  function handleClear() {
    if (!confirmClear) {
      setConfirmClear(true);
      return;
    }
    onClearMemory();
    setConfirmClear(false);
    setOpen(false);
  }

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => { setOpen((v) => !v); setConfirmClear(false); }}
        title="Settings"
        aria-label="Open settings"
        className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-colors"
      >
        <Settings size={18} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-72 bg-white rounded-2xl shadow-lg border border-gray-100 p-4 z-50">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Settings</h3>

          {/* Voice selection */}
          {isSupported && availableVoices.length > 0 && (
            <div className="mb-4">
              <label className="block text-xs font-medium text-gray-500 mb-1.5">Voice</label>
              <select
                value={selectedVoiceIndex}
                onChange={(e) => setSelectedVoiceIndex(Number(e.target.value))}
                className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 text-gray-700
                  focus:outline-none focus:ring-2 focus:ring-blue-300 bg-white"
              >
                {availableVoices.map((v, i) => (
                  <option key={i} value={i}>
                    {v.name} ({v.lang})
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Speech rate */}
          {isSupported && (
            <div className="mb-5">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-gray-500">Speech Rate</label>
                <span className="text-xs text-gray-400 tabular-nums">{rate.toFixed(1)}×</span>
              </div>
              <input
                type="range"
                min={0.7}
                max={1.3}
                step={0.05}
                value={rate}
                onChange={(e) => setRate(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
              <div className="flex justify-between text-xs text-gray-300 mt-1">
                <span>0.7×</span>
                <span>1.0×</span>
                <span>1.3×</span>
              </div>
            </div>
          )}

          {/* Divider */}
          <div className="border-t border-gray-100 pt-3">
            <button
              onClick={handleClear}
              className={`w-full text-sm px-3 py-2 rounded-lg transition-colors font-medium ${
                confirmClear
                  ? 'bg-red-500 text-white hover:bg-red-600'
                  : 'text-red-500 hover:bg-red-50'
              }`}
            >
              {confirmClear ? 'Tap again to confirm clear' : 'Clear all memory'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── History tab ─────────────────────────────────────────────────────────────

function HistoryList({ entries }: { entries: DiaryEntry[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const sorted = [...entries].reverse();

  if (sorted.length === 0) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 text-center">
        <p className="text-3xl mb-3">📖</p>
        <p className="text-sm text-gray-400">No diary entries yet.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {sorted.map((entry) => (
        <div
          key={entry.id}
          className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
        >
          <button
            className="w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-gray-50 transition-colors"
            onClick={() => setExpanded(expanded === entry.id ? null : entry.id)}
          >
            <div className="flex-1 min-w-0">
              <p className="text-xs text-gray-400">
                {new Date(entry.date).toLocaleDateString('en-US', {
                  weekday: 'short', month: 'short', day: 'numeric', year: 'numeric',
                })}
              </p>
              <p className="text-sm text-gray-600 truncate mt-0.5">{entry.rawText}</p>
            </div>
            {entry.feedback && (
              <div className="shrink-0 flex items-center justify-center w-10 h-10 rounded-full bg-blue-50 text-blue-600 text-sm font-bold">
                {entry.feedback.overallScore}
              </div>
            )}
            <span className="text-gray-300 text-xs">{expanded === entry.id ? '▲' : '▼'}</span>
          </button>

          {expanded === entry.id && entry.feedback && (
            <div className="px-5 pb-5 border-t border-gray-50">
              <FeedbackPanel feedback={entry.feedback} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Inner app (needs VoiceProvider in scope) ─────────────────────────────────

function AppInner() {
  const [diaryText, setDiaryText] = useState('');
  const [activeTab, setActiveTab] = useState<ActiveTab>('feedback');

  const { store, applyFeedback, markMastered, clearAll } = useMemory();
  const { result: feedbackResult, isLoading, error, analyze } = useFeedback();

  async function handleGetFeedback() {
    if (!diaryText.trim() || isLoading) return;
    const feedback = await analyze(diaryText, store);
    if (feedback) {
      applyFeedback(diaryText, feedback);
      setActiveTab('feedback');
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-20 bg-white/80 backdrop-blur border-b border-gray-100">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center gap-4">
          <h1 className="text-base font-bold text-gray-800 shrink-0">
            英語日記
            <span className="ml-2 text-blue-500 font-normal text-sm">eigo-diary</span>
          </h1>
          <div className="flex-1" />
          <SettingsPopover onClearMemory={clearAll} />
        </div>
      </header>

      {/* Body: sidebar + main */}
      <div className="flex-1 max-w-6xl w-full mx-auto px-6 py-6 flex gap-6">

        {/* Left sidebar — MemoryPanel */}
        <aside
          className="shrink-0 flex flex-col gap-4 self-start sticky top-[61px]"
          style={{ width: 280 }}
        >
          <MemoryPanel store={store} onMarkMastered={markMastered} />
        </aside>

        {/* Main content */}
        <main className="flex-1 min-w-0 space-y-5">
          <DiaryInput
            value={diaryText}
            onChange={setDiaryText}
            onSubmit={handleGetFeedback}
            isLoading={isLoading}
          />

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-2xl p-4 text-sm text-red-700 leading-relaxed">
              <strong className="font-semibold">Something went wrong:</strong>{' '}
              <span className="font-mono text-xs">{error}</span>
            </div>
          )}

          {/* Tab bar — only show once there's feedback or history */}
          {(feedbackResult || store.diaryHistory.length > 0) && (
            <div className="flex gap-1 bg-gray-100 p-1 rounded-xl w-fit">
              {(['feedback', 'history'] as ActiveTab[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors capitalize ${
                    activeTab === t
                      ? 'bg-white text-gray-800 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {t}
                  {t === 'history' && store.diaryHistory.length > 0 && (
                    <span className="ml-1.5 text-xs bg-gray-200 text-gray-600 px-1.5 py-0.5 rounded-full">
                      {store.diaryHistory.length}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}

          {activeTab === 'feedback' && feedbackResult && (
            <FeedbackPanel feedback={feedbackResult} />
          )}

          {activeTab === 'history' && (
            <HistoryList entries={store.diaryHistory} />
          )}
        </main>
      </div>
    </div>
  );
}

// ── Root export ──────────────────────────────────────────────────────────────

export default function App() {
  return (
    <VoiceProvider>
      <AppInner />
    </VoiceProvider>
  );
}

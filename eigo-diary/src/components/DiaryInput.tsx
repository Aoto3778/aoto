interface DiaryInputProps {
  value: string;
  onChange: (text: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

export default function DiaryInput({ value, onChange, onSubmit, isLoading }: DiaryInputProps) {
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && value.trim() && !isLoading) {
      onSubmit();
    }
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Today's Entry</h2>
        <span className="text-sm text-gray-400">{formatDate(new Date())}</span>
      </div>

      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Write about your day in English... (any length)"
        disabled={isLoading}
        rows={8}
        className="w-full resize-none rounded-xl border border-gray-200 p-4 text-gray-700 placeholder-gray-300
          focus:outline-none focus:ring-2 focus:ring-blue-300 focus:border-transparent
          disabled:bg-gray-50 disabled:text-gray-400 transition-colors text-sm leading-relaxed"
      />

      <div className="flex items-center justify-between mt-3">
        <span className="text-xs text-gray-400">
          {value.length} characters
          <span className="ml-2 text-gray-300">· ⌘↵ to submit</span>
        </span>

        <button
          onClick={onSubmit}
          disabled={isLoading || !value.trim()}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-500 text-white text-sm font-medium
            hover:bg-blue-600 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed
            transition-colors focus:outline-none focus:ring-2 focus:ring-blue-300"
        >
          {isLoading ? (
            <>
              <span>Analyzing</span>
              <span className="flex gap-0.5">
                <span className="animate-bounce [animation-delay:0ms] inline-block w-1 h-1 rounded-full bg-white/70" />
                <span className="animate-bounce [animation-delay:150ms] inline-block w-1 h-1 rounded-full bg-white/70" />
                <span className="animate-bounce [animation-delay:300ms] inline-block w-1 h-1 rounded-full bg-white/70" />
              </span>
            </>
          ) : (
            'Get Feedback'
          )}
        </button>
      </div>
    </div>
  );
}

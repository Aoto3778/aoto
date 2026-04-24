import type { FeedbackResult } from '../types';
import VoiceButton from './VoiceButton';

interface FeedbackPanelProps {
  feedback: FeedbackResult;
}

const CATEGORY_COLORS: Record<string, string> = {
  preposition: 'bg-purple-100 text-purple-700',
  tense: 'bg-yellow-100 text-yellow-700',
  article: 'bg-blue-100 text-blue-700',
  uncountable: 'bg-orange-100 text-orange-700',
  'word-order': 'bg-pink-100 text-pink-700',
  other: 'bg-gray-100 text-gray-600',
};

const CHALLENGE_TYPE_COLORS: Record<string, string> = {
  grammar: 'bg-blue-100 text-blue-700',
  idiom: 'bg-green-100 text-green-700',
  vocabulary: 'bg-amber-100 text-amber-700',
};

function CircularScore({ score }: { score: number }) {
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(Math.max(score, 0), 100);
  const offset = circumference - (progress / 100) * circumference;

  const color =
    progress >= 80 ? '#22c55e' : progress >= 65 ? '#3b82f6' : '#f59e0b';

  return (
    <div className="relative inline-flex items-center justify-center w-24 h-24">
      <svg width="96" height="96" className="-rotate-90">
        <circle
          cx="48"
          cy="48"
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="8"
        />
        <circle
          cx="48"
          cy="48"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.6s ease' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-2xl font-bold text-gray-800">{score}</span>
        <span className="text-xs text-gray-400">/ 100</span>
      </div>
    </div>
  );
}

export default function FeedbackPanel({ feedback }: FeedbackPanelProps) {
  const {
    correctedText,
    grammarErrors,
    idiomSuggestions,
    newChallenges,
    overallScore,
    encouragement,
  } = feedback;

  return (
    <div className="space-y-5">
      {/* Score + encouragement */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 flex gap-6 items-start">
        <CircularScore score={overallScore} />
        <div className="flex-1">
          <h3 className="font-semibold text-gray-700 mb-2">Overall Score</h3>
          <p className="text-sm text-gray-600 leading-relaxed bg-blue-50 border border-blue-100 rounded-xl p-3">
            {encouragement}
          </p>
        </div>
      </div>

      {/* Corrected text */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-gray-700">Corrected Version</h3>
          <VoiceButton text={correctedText} label="Listen" />
        </div>
        <p className="text-sm text-gray-600 leading-relaxed whitespace-pre-wrap">{correctedText}</p>
      </div>

      {/* Grammar errors */}
      {grammarErrors.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="font-semibold text-gray-700 mb-4">
            Grammar Corrections
            <span className="ml-2 text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full font-normal">
              {grammarErrors.length}
            </span>
          </h3>
          <div className="space-y-3">
            {grammarErrors.map((err, i) => (
              <div key={i} className="rounded-xl border border-gray-100 bg-gray-50 p-4">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span className="text-sm line-through text-red-500">{err.original}</span>
                  <span className="text-gray-400 text-xs">→</span>
                  <span className="text-sm font-medium text-green-600">{err.corrected}</span>
                  <span
                    className={`ml-auto text-xs px-2 py-0.5 rounded-full font-medium ${
                      CATEGORY_COLORS[err.category] ?? CATEGORY_COLORS.other
                    }`}
                  >
                    {err.category}
                  </span>
                </div>
                <p className="text-xs text-gray-500 leading-relaxed">{err.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Idiom suggestions */}
      {idiomSuggestions.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="font-semibold text-gray-700 mb-4">Useful Expressions</h3>
          <div className="space-y-3">
            {idiomSuggestions.map((idiom, i) => (
              <div key={i} className="rounded-xl border border-green-100 bg-green-50 p-4">
                <div className="flex items-start justify-between gap-2 mb-1">
                  <span className="text-sm font-semibold text-green-700">"{idiom.expression}"</span>
                  <VoiceButton text={idiom.example} rate={0.85} />
                </div>
                <p className="text-xs text-gray-500 mb-1">{idiom.usage}</p>
                <p className="text-xs text-gray-600 italic">e.g. "{idiom.example}"</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* New challenges */}
      {newChallenges.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="font-semibold text-gray-700 mb-4">New Challenges Added</h3>
          <div className="space-y-2">
            {newChallenges.map((c, i) => (
              <div key={i} className="flex items-start gap-3 py-2 border-b border-gray-50 last:border-0">
                <span
                  className={`shrink-0 text-xs px-2 py-0.5 rounded-full font-medium mt-0.5 ${
                    CHALLENGE_TYPE_COLORS[c.type] ?? 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {c.type}
                </span>
                <div>
                  <p className="text-sm font-medium text-gray-700">{c.item}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{c.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

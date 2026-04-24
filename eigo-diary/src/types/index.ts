export interface GrammarError {
  original: string;
  corrected: string;
  explanation: string;
  category: string;
}

export interface IdiomSuggestion {
  expression: string;
  usage: string;
  example: string;
}

export interface Challenge {
  type: 'grammar' | 'idiom' | 'vocabulary';
  item: string;
  reason: string;
}

export interface FeedbackResult {
  correctedText: string;
  grammarErrors: GrammarError[];
  idiomSuggestions: IdiomSuggestion[];
  newChallenges: Challenge[];
  overallScore: number;
  encouragement: string;
}

export interface DiaryEntry {
  id: string;
  date: string;
  rawText: string;
  feedback: FeedbackResult | null;
}

export interface ErrorPattern {
  category: string;
  count: number;
  lastSeen: string;
  examples: string[];
}

export interface CurrentChallenge {
  type: 'grammar' | 'idiom' | 'vocabulary';
  item: string;
  addedDate: string;
  masteredDate?: string;
}

export interface MemoryStore {
  errorPatterns: ErrorPattern[];
  currentChallenges: CurrentChallenge[];
  masteredItems: string[];
  diaryHistory: DiaryEntry[];
}

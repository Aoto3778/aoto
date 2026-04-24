import type { FeedbackResult, MemoryStore } from '../types';

const STORAGE_KEY = 'eigo-diary-memory';

const defaultStore = (): MemoryStore => ({
  errorPatterns: [],
  currentChallenges: [],
  masteredItems: [],
  diaryHistory: [],
});

export function loadMemory(): MemoryStore {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultStore();
    return JSON.parse(raw) as MemoryStore;
  } catch {
    return defaultStore();
  }
}

export function saveMemory(store: MemoryStore): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

export function updateMemoryFromFeedback(
  store: MemoryStore,
  feedback: FeedbackResult,
): MemoryStore {
  const now = new Date().toISOString();

  // Merge grammar error patterns
  const patterns = [...store.errorPatterns];
  for (const error of feedback.grammarErrors) {
    const existing = patterns.find((p) => p.category === error.category);
    if (existing) {
      existing.count += 1;
      existing.lastSeen = now;
      if (!existing.examples.includes(error.original)) {
        existing.examples.push(error.original);
      }
    } else {
      patterns.push({
        category: error.category,
        count: 1,
        lastSeen: now,
        examples: [error.original],
      });
    }
  }

  // Add new challenges, skipping items already tracked or mastered
  const trackedItems = new Set([
    ...store.currentChallenges.map((c) => c.item),
    ...store.masteredItems,
  ]);
  const newChallenges = feedback.newChallenges
    .filter((c) => !trackedItems.has(c.item))
    .map((c) => ({ ...c, addedDate: now }));

  return {
    ...store,
    errorPatterns: patterns,
    currentChallenges: [...store.currentChallenges, ...newChallenges],
  };
}

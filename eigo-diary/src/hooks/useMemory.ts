import { useState, useCallback } from 'react';
import { loadMemory, saveMemory, updateMemoryFromFeedback } from '../lib/storage';
import type { FeedbackResult, MemoryStore } from '../types';

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function useMemory() {
  const [store, setStore] = useState<MemoryStore>(loadMemory);

  const applyFeedback = useCallback((rawText: string, result: FeedbackResult) => {
    setStore((prev) => {
      const updated = updateMemoryFromFeedback(prev, result);
      const next: MemoryStore = {
        ...updated,
        diaryHistory: [
          ...updated.diaryHistory,
          { id: makeId(), date: new Date().toISOString(), rawText, feedback: result },
        ],
      };
      saveMemory(next);
      return next;
    });
  }, []);

  const markMastered = useCallback((item: string) => {
    setStore((prev) => {
      const next: MemoryStore = {
        ...prev,
        currentChallenges: prev.currentChallenges.filter((c) => c.item !== item),
        masteredItems: [...prev.masteredItems, item],
      };
      saveMemory(next);
      return next;
    });
  }, []);

  const clearAll = useCallback(() => {
    const empty: MemoryStore = {
      errorPatterns: [],
      currentChallenges: [],
      masteredItems: [],
      diaryHistory: [],
    };
    saveMemory(empty);
    setStore(empty);
  }, []);

  return { store, applyFeedback, markMastered, clearAll };
}

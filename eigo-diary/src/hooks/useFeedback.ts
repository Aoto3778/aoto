import { useState, useCallback } from 'react';
import { analyzeDiary } from '../lib/claude';
import type { FeedbackResult, MemoryStore } from '../types';

export function useFeedback() {
  const [result, setResult] = useState<FeedbackResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (text: string, store: MemoryStore): Promise<FeedbackResult | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const feedback = await analyzeDiary(text, store);
      setResult(feedback);
      return feedback;
    } catch (err) {
      setError((err as Error).message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, isLoading, error, analyze, reset };
}

import { useState, useEffect, useCallback, useRef } from 'react';

const isSupported =
  typeof window !== 'undefined' &&
  'speechSynthesis' in window &&
  'SpeechSynthesisUtterance' in window;

function pickDefaultVoiceIndex(voices: SpeechSynthesisVoice[]): number {
  // Prefer female-named en-US voices, then en-GB, then any English voice
  const priorities = [
    (v: SpeechSynthesisVoice) => /en-US/i.test(v.lang) && /female|zira|samantha|victoria/i.test(v.name),
    (v: SpeechSynthesisVoice) => /en-GB/i.test(v.lang) && /female|serena|kate/i.test(v.name),
    (v: SpeechSynthesisVoice) => /en-US/i.test(v.lang),
    (v: SpeechSynthesisVoice) => /en-GB/i.test(v.lang),
    (v: SpeechSynthesisVoice) => /^en/i.test(v.lang),
  ];
  for (const test of priorities) {
    const idx = voices.findIndex(test);
    if (idx !== -1) return idx;
  }
  return 0;
}

export interface VoiceHook {
  speak: (text: string, options?: { rate?: number; pitch?: number }) => void;
  stop: () => void;
  isSpeaking: boolean;
  isSupported: boolean;
  availableVoices: SpeechSynthesisVoice[];
  selectedVoiceIndex: number;
  setSelectedVoiceIndex: (i: number) => void;
}

export function useVoice(): VoiceHook {
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoiceIndex, setSelectedVoiceIndex] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  // Track the current utterance so we can cancel it precisely
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Load voices — voiceschanged fires asynchronously in most browsers
  useEffect(() => {
    if (!isSupported) return;

    function loadVoices() {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length === 0) return;
      setAvailableVoices(voices);
      setSelectedVoiceIndex((prev) =>
        // Only auto-select on the first load (when we're still at 0 with no voices)
        prev === 0 ? pickDefaultVoiceIndex(voices) : prev,
      );
    }

    loadVoices();
    window.speechSynthesis.addEventListener('voiceschanged', loadVoices);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', loadVoices);
  }, []);

  const stop = useCallback(() => {
    if (!isSupported) return;
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    utteranceRef.current = null;
  }, []);

  const speak = useCallback(
    (text: string, options?: { rate?: number; pitch?: number }) => {
      if (!isSupported || !text.trim()) return;

      // Cancel any in-progress speech first
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = options?.rate ?? 0.9;
      utterance.pitch = options?.pitch ?? 1.0;

      if (availableVoices[selectedVoiceIndex]) {
        utterance.voice = availableVoices[selectedVoiceIndex];
      }

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => {
        setIsSpeaking(false);
        utteranceRef.current = null;
      };
      utterance.onerror = () => {
        setIsSpeaking(false);
        utteranceRef.current = null;
      };

      utteranceRef.current = utterance;
      window.speechSynthesis.speak(utterance);
    },
    [availableVoices, selectedVoiceIndex],
  );

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (isSupported) window.speechSynthesis.cancel();
    };
  }, []);

  return {
    speak,
    stop,
    isSpeaking,
    isSupported,
    availableVoices,
    selectedVoiceIndex,
    setSelectedVoiceIndex,
  };
}

import { createContext, useContext, useState, type ReactNode } from 'react';
import { useVoice } from '../hooks/useVoice';
import type { VoiceHook } from '../hooks/useVoice';

interface VoiceSettings extends Omit<VoiceHook, 'speak'> {
  speak: (text: string, options?: { rate?: number; pitch?: number }) => void;
  rate: number;
  setRate: (r: number) => void;
}

const VoiceContext = createContext<VoiceSettings | null>(null);

export function VoiceProvider({ children }: { children: ReactNode }) {
  const hook = useVoice();
  const [rate, setRate] = useState(0.9);

  function speak(text: string, options?: { rate?: number; pitch?: number }) {
    hook.speak(text, { rate: options?.rate ?? rate, pitch: options?.pitch });
  }

  return (
    <VoiceContext.Provider value={{ ...hook, speak, rate, setRate }}>
      {children}
    </VoiceContext.Provider>
  );
}

export function useVoiceSettings(): VoiceSettings {
  const ctx = useContext(VoiceContext);
  if (!ctx) throw new Error('useVoiceSettings must be used inside VoiceProvider');
  return ctx;
}

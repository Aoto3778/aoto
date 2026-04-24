import { Volume2, VolumeX } from 'lucide-react';
import { useVoice } from '../hooks/useVoice';

interface VoiceButtonProps {
  text: string;
  rate?: number;
  pitch?: number;
  /** Optional label shown beside the icon for accessibility */
  label?: string;
  className?: string;
}

export default function VoiceButton({
  text,
  rate,
  pitch,
  label,
  className = '',
}: VoiceButtonProps) {
  const { speak, stop, isSpeaking, isSupported } = useVoice();

  if (!isSupported) {
    return (
      <button
        disabled
        title="Text-to-speech is not supported in this browser"
        className={`inline-flex items-center gap-1 px-2 py-1 rounded text-gray-400 cursor-not-allowed opacity-50 ${className}`}
        aria-label="Text-to-speech unavailable"
      >
        <VolumeX size={16} />
        {label && <span className="text-xs">{label}</span>}
      </button>
    );
  }

  function handleClick() {
    if (isSpeaking) {
      stop();
    } else {
      speak(text, { rate, pitch });
    }
  }

  return (
    <button
      onClick={handleClick}
      title={isSpeaking ? 'Stop speaking' : 'Read aloud'}
      aria-label={isSpeaking ? 'Stop speaking' : 'Read aloud'}
      aria-pressed={isSpeaking}
      className={`relative inline-flex items-center gap-1 px-2 py-1 rounded transition-colors
        ${isSpeaking
          ? 'text-blue-600 hover:text-blue-700'
          : 'text-gray-500 hover:text-gray-700'}
        ${className}`}
    >
      {/* Animated pulse ring visible only while speaking */}
      {isSpeaking && (
        <span
          aria-hidden="true"
          className="absolute inset-0 rounded animate-ping opacity-20 bg-blue-400"
        />
      )}
      <Volume2 size={16} className="relative" />
      {label && <span className="text-xs relative">{label}</span>}
    </button>
  );
}

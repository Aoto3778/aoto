import Anthropic from '@anthropic-ai/sdk';
import type { FeedbackResult, MemoryStore } from '../types';

export const anthropic = new Anthropic({
  apiKey: import.meta.env.VITE_ANTHROPIC_API_KEY,
  dangerouslyAllowBrowser: true,
});

function buildSystemPrompt(memoryStore: MemoryStore): string {
  const errorPatternsSummary =
    memoryStore.errorPatterns.length === 0
      ? 'None recorded yet.'
      : memoryStore.errorPatterns
          .map((p) => `- ${p.category} (seen ${p.count}x, e.g. "${p.examples[0]}")`)
          .join('\n');

  const challengesSummary =
    memoryStore.currentChallenges.length === 0
      ? 'None assigned yet.'
      : memoryStore.currentChallenges
          .map((c) => `- [${c.type}] ${c.item}`)
          .join('\n');

  return `You are a warm, encouraging English teacher helping a Japanese adult learner improve their English through daily journaling.

Your job is to analyze their diary entry and return feedback in valid JSON only — no markdown, no preamble.

The user's current known weaknesses are:
${errorPatternsSummary}

Their current grammar/idiom challenges are:
${challengesSummary}

Return this exact JSON structure:
{
  "correctedText": "full corrected diary text",
  "grammarErrors": [
    {
      "original": "the text they wrote",
      "corrected": "the corrected version",
      "explanation": "why this is wrong and the rule",
      "category": "preposition|tense|article|uncountable|word-order|other"
    }
  ],
  "idiomSuggestions": [
    {
      "expression": "idiom or phrase",
      "usage": "when/how to use it",
      "example": "example sentence related to their diary content"
    }
  ],
  "newChallenges": [
    {
      "type": "grammar|idiom|vocabulary",
      "item": "the thing to practice",
      "reason": "why this suits them based on their errors"
    }
  ],
  "overallScore": 75,
  "encouragement": "2-3 sentences of specific, genuine encouragement referencing their content"
}

Rules:
- Preserve the user's voice and content; only fix clear errors
- Suggest 2-3 idioms that naturally fit what they wrote about
- Add 1-2 new challenges only if genuinely needed based on recurring patterns
- Score honestly but warmly (60-90 range for learners)
- Encouragement must be specific, not generic ("I liked how you described X")`;
}

export async function analyzeDiary(
  diaryText: string,
  memoryStore: MemoryStore,
): Promise<FeedbackResult> {
  const response = await anthropic.messages.create({
    model: 'claude-opus-4-5',
    max_tokens: 4096,
    system: buildSystemPrompt(memoryStore),
    messages: [{ role: 'user', content: diaryText }],
  });

  const firstBlock = response.content[0];
  if (firstBlock.type !== 'text') {
    throw new Error(`Unexpected response block type: ${firstBlock.type}`);
  }

  // Strip accidental markdown fences the model might add despite instructions
  const raw = firstBlock.text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `Claude returned invalid JSON. Parse error: ${(err as Error).message}\n\nRaw response:\n${raw}`,
    );
  }

  const result = parsed as FeedbackResult;

  if (
    typeof result.correctedText !== 'string' ||
    !Array.isArray(result.grammarErrors) ||
    !Array.isArray(result.idiomSuggestions) ||
    !Array.isArray(result.newChallenges) ||
    typeof result.overallScore !== 'number' ||
    typeof result.encouragement !== 'string'
  ) {
    throw new Error(
      `Claude response is missing required FeedbackResult fields.\n\nParsed value:\n${JSON.stringify(parsed, null, 2)}`,
    );
  }

  return result;
}

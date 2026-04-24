/**
 * End-to-end smoke test for eigo-diary.
 * Intercepts the Anthropic API call and returns a realistic mock,
 * then verifies every UI section and localStorage.
 */
import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';

const DIARY_TEXT = `Today I went to the park with my friend.
We are very enjoy the weather.
I buyed some coffee and we talked about many informations.`;

const MOCK_FEEDBACK = {
  correctedText:
    'Today I went to the park with my friend. We really enjoyed the weather. I bought some coffee and we talked about a lot of things.',
  grammarErrors: [
    {
      original: 'We are very enjoy the weather',
      corrected: 'We really enjoyed the weather',
      explanation:
        "Use simple past tense 'enjoyed'. 'Are enjoy' mixes the present 'be' verb with a base verb — the correct form is just 'enjoyed'.",
      category: 'tense',
    },
    {
      original: 'I buyed some coffee',
      corrected: 'I bought some coffee',
      explanation:
        "'Buy' is an irregular verb. The simple past form is 'bought', not 'buyed'.",
      category: 'tense',
    },
    {
      original: 'many informations',
      corrected: 'a lot of information',
      explanation:
        "'Information' is an uncountable noun in English and cannot be pluralised as 'informations'. Use 'information' or 'a lot of information'.",
      category: 'uncountable',
    },
  ],
  idiomSuggestions: [
    {
      expression: 'catch up',
      usage: 'Use when meeting someone to share recent news after time apart.',
      example: 'We met at the park to catch up over coffee.',
    },
    {
      expression: 'soak up the sun',
      usage: 'Use to describe enjoying warm, sunny weather outdoors.',
      example: 'We sat on the bench and soaked up the sun.',
    },
  ],
  newChallenges: [
    {
      type: 'grammar',
      item: 'Irregular past tense verbs (buy → bought)',
      reason: "You used 'buyed' — irregular past forms need memorisation.",
    },
    {
      type: 'grammar',
      item: 'Uncountable nouns (information, advice, news)',
      reason: "You wrote 'informations' — these nouns have no plural form.",
    },
  ],
  overallScore: 72,
  encouragement:
    "Great job writing about your day! Your story is clear and engaging. You already used 'went' correctly — keep building on that and focus on irregular past tense verbs like 'bought' next.",
};

// Anthropic messages API response envelope
const MOCK_API_RESPONSE = {
  id: 'msg_test_eigo_diary_01',
  type: 'message',
  role: 'assistant',
  model: 'claude-opus-4-5',
  content: [{ type: 'text', text: JSON.stringify(MOCK_FEEDBACK) }],
  stop_reason: 'end_turn',
  usage: { input_tokens: 312, output_tokens: 418 },
};

// ── helpers ──────────────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function check(label, value) {
  if (value) {
    console.log(`  ✅ ${label}`);
    passed++;
  } else {
    console.error(`  ❌ FAIL: ${label}`);
    failed++;
  }
}

// ── main ─────────────────────────────────────────────────────────────────────

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

// Capture all console messages from the page
const consoleMessages = [];
page.on('console', (msg) => {
  const type = msg.type();
  const text = msg.text();
  consoleMessages.push({ type, text });
  if (type === 'error') console.log(`  [browser ${type}] ${text}`);
});

page.on('pageerror', (err) => {
  console.error(`  [page error] ${err.message}`);
});

// Intercept Anthropic API call
await page.route('**/v1/messages', async (route) => {
  console.log('  → Intercepted POST /v1/messages — returning mock response');
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(MOCK_API_RESPONSE),
  });
});

// ── Step 1: Load app ─────────────────────────────────────────────────────────
console.log('\n🔵 Step 1: Load app');
await page.goto('http://localhost:5173/');
await page.waitForLoadState('networkidle');

check('Page title contains eigo-diary', (await page.title()).includes('eigo-diary'));
check('DiaryInput textarea visible', await page.locator('textarea').isVisible());
check('MemoryPanel sidebar visible', await page.locator('text=Your Journey').isVisible());
check(
  'Get Feedback button disabled (empty textarea)',
  await page.locator('button:has-text("Get Feedback")').isDisabled(),
);

// ── Step 2: Type diary entry ─────────────────────────────────────────────────
console.log('\n🔵 Step 2: Type diary entry');
await page.locator('textarea').fill(DIARY_TEXT);
await page.waitForTimeout(100);

const charCount = await page.locator('text=/\\d+ characters/').textContent();
check(`Character count shows (got: "${charCount?.trim()}")`, charCount && parseInt(charCount) > 0);
check(
  'Get Feedback button enabled after typing',
  await page.locator('button:has-text("Get Feedback")').isEnabled(),
);

// ── Step 3: Submit and wait for FeedbackPanel ────────────────────────────────
console.log('\n🔵 Step 3: Submit → wait for API response');
await page.locator('button:has-text("Get Feedback")').click();

// Loading dots should appear briefly
const loadingVisible = await page.locator('text=Analyzing').isVisible().catch(() => false);
console.log(`  ℹ️  Loading state captured: ${loadingVisible}`);

// Wait for feedback panel to appear (score circle + corrected version)
await page.waitForSelector('text=Corrected Version', { timeout: 10000 });
console.log('  → FeedbackPanel appeared');

// ── Step 4: Verify FeedbackPanel sections ────────────────────────────────────
console.log('\n🔵 Step 4: Verify FeedbackPanel sections');

// Score circle
const scoreText = await page.locator('text=Overall Score').isVisible();
check('Overall Score section visible', scoreText);
const score = await page.locator('text=72').first().isVisible();
check('overallScore value 72 rendered', score);

// Corrected text
check('Corrected Version heading visible', await page.locator('text=Corrected Version').isVisible());
const correctedSnippet = await page.locator('text=really enjoyed').first().isVisible();
check('Corrected text contains "really enjoyed"', correctedSnippet);

// VoiceButton on corrected text
const listenBtn = page.locator('button[aria-label="Read aloud"]').first();
check('VoiceButton (Listen) present on corrected text', await listenBtn.isVisible());

// Grammar errors
check('Grammar Corrections heading visible', await page.locator('text=Grammar Corrections').isVisible());
check('Error count badge shows 3', await page.locator('text="3"').first().isVisible());
check('"buyed" shown with strikethrough', await page.locator('text=buyed').first().isVisible());
check('"bought" shown as correction', await page.locator('text=bought').first().isVisible());
check('tense category badge present', await page.locator('text=tense').first().isVisible());
check('uncountable category badge present', await page.locator('text=uncountable').first().isVisible());

// Idiom suggestions
check('Useful Expressions heading visible', await page.locator('text=Useful Expressions').isVisible());
check('"catch up" idiom card rendered', await page.locator('text=catch up').first().isVisible());
check('"soak up the sun" idiom card rendered', await page.locator('text=soak up the sun').isVisible());

// New challenges
check('New Challenges Added heading visible', await page.locator('text=New Challenges Added').isVisible());
check('Grammar challenge badge visible', await page.locator('text=grammar').first().isVisible());

// Encouragement
check('Encouragement text visible', await page.locator('text=Great job writing').isVisible());

// Feedback/history tab bar — history button text is "history1" (count inlined)
check('Feedback tab active', await page.locator('button').filter({ hasText: /^feedback/i }).isVisible());
check('History tab appears', await page.locator('button').filter({ hasText: /^history/i }).isVisible());

// ── Step 5: Check MemoryPanel updated ────────────────────────────────────────
console.log('\n🔵 Step 5: Verify MemoryPanel sidebar updated');

check('Error Patterns section visible', await page.locator('text=Error Patterns').isVisible());
check('tense bar rendered in sidebar', await page.locator('text=tense').nth(1).isVisible().catch(() =>
  page.locator('aside text=tense').isVisible()
));
check('Current challenge appears in sidebar', await page.locator('aside').locator('text=/irregular past/i').isVisible().catch(() =>
  page.locator('aside').locator('text=grammar').isVisible()
));

// ── Step 6: Check localStorage ───────────────────────────────────────────────
console.log('\n🔵 Step 6: Verify localStorage');

const stored = await page.evaluate(() => {
  const raw = localStorage.getItem('eigo-diary-memory');
  return raw ? JSON.parse(raw) : null;
});

check('localStorage key "eigo-diary-memory" exists', stored !== null);
check('errorPatterns has tense entry', stored?.errorPatterns?.some((p) => p.category === 'tense'));
check('errorPatterns has uncountable entry', stored?.errorPatterns?.some((p) => p.category === 'uncountable'));
check('tense count >= 2', stored?.errorPatterns?.find((p) => p.category === 'tense')?.count >= 2);
check('currentChallenges has 2 entries', stored?.currentChallenges?.length === 2);
check('diaryHistory has 1 entry', stored?.diaryHistory?.length === 1);
check('diaryHistory entry has feedback', stored?.diaryHistory?.[0]?.feedback !== null);
check('diaryHistory entry score is 72', stored?.diaryHistory?.[0]?.feedback?.overallScore === 72);

// Raw dump for inspection
console.log('\n  localStorage snapshot:');
console.log('    errorPatterns:', JSON.stringify(stored?.errorPatterns?.map(p => `${p.category}×${p.count}`)));
console.log('    currentChallenges:', stored?.currentChallenges?.length);
console.log('    diaryHistory:', stored?.diaryHistory?.length, 'entries');

// ── Step 7: History tab ──────────────────────────────────────────────────────
console.log('\n🔵 Step 7: Switch to History tab');
await page.locator('button:has-text("History")').click();
await page.waitForTimeout(300);

check('History entry row visible', await page.locator('text=Today I went').first().isVisible());

// Expand the entry — click the row <button>, not the inner text node
await page.locator('button').filter({ hasText: 'Today I went' }).first().click();
await page.waitForTimeout(400);
check('Expanded history shows FeedbackPanel', await page.locator('text=Corrected Version').isVisible());

// ── Step 8: Settings popover ─────────────────────────────────────────────────
console.log('\n🔵 Step 8: Settings popover');
await page.locator('button[aria-label="Open settings"]').click();
await page.waitForTimeout(200);

check('Settings popover opens', await page.locator('text=Speech Rate').isVisible());

const voiceSelectExists = await page.locator('select').isVisible().catch(() => false);
console.log(`  ℹ️  Voice dropdown visible: ${voiceSelectExists} (headless Chrome may have no voices)`);

// Rate slider
const slider = page.locator('input[type="range"]');
check('Rate slider visible', await slider.isVisible());
// Use Playwright's fill() — raw DOM events are ignored by React's synthetic event system
await slider.fill('1.1');
await page.waitForTimeout(100);
const rateLabel = await page.locator('text=/1\\.1×/').isVisible();
check('Rate slider updates to 1.1×', rateLabel);

// Close popover
await page.keyboard.press('Escape');
await page.locator('body').click();

// ── Summary ──────────────────────────────────────────────────────────────────
console.log('\n' + '─'.repeat(50));
const total = passed + failed;
console.log(`Result: ${passed}/${total} checks passed${failed > 0 ? ` (${failed} FAILED)` : ' ✅'}`);

if (consoleMessages.filter((m) => m.type === 'error').length > 0) {
  console.log('\nBrowser console errors:');
  consoleMessages.filter((m) => m.type === 'error').forEach((m) => console.log('  ', m.text));
}

await browser.close();
process.exit(failed > 0 ? 1 : 0);

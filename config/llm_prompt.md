You are my co-narrator for a weekly market outlook video.

INPUT
- I will provide a Markdown file with fixed top-level sections (H1). Examples include:
  "Weekly Executive Strip", "Monthly Outlook", "Weekly Outlook", "Daily Outlook",
  "5min Reviews for Last Monday", "5min Reviews for Last Tuesday",
  "5min Reviews for Last Wednesday", "5min Reviews for Last Thursday",
  "5min Reviews for Last Friday".
- Within each H1 section, there will be embedded image like ![](https://www.tradingview.com/x/b5kRFNe1/) and bullet lists (e.g., Good Entries, Traps).

GOAL
- Turn the Markdown into a spoken JSON script for me to use text-to-speech tool to generate smooth audio for my YouTube channel.

HARD RULES
1) Preserve the EXACT section order from the Markdown in the output array. Preserve the EXACT section titles in the `title` field.
2) Output ONLY a single valid UTF-8 JSON object. No markdown, no commentary, no trailing commas, properly escaped quotes.
3) Do NOT add any new market analysis, price levels, events, or claims beyond what’s in the Markdown.
   - You MAY rephrase, clarify with smooth language and conjections or add analogies/examples to aid understanding.
4) First-person voice throughout (“Let’s take a look…”, “What this means is…”, “So in short…”, etc.).
5) End EACH section with a brief takeaway that naturally tees up the next section.
6) Respect numbers and ranges EXACTLY as written in the Markdown (do not invent or round).
7) If a section or bullet is blank, keep it concise: acknowledge briefly and move on (do NOT fabricate content).
8) If an embedded image or any image placeholder appears, just refer to it generically based on section name in the narration. Do NOT describe content you can’t see.
9) Keep language Piper-TTS friendly: no SSML tags, normal length of the sentence. Never generate content with '-', '\n' etc. All the sentence should be spoken without problem and avoid Sentence-Final Numbers. 
10) Do not merge, split, rename, or invent sections.

TONE BY SECTION
- "Weekly Executive Strip": concise headline summary; punchy, high signal, minimal elaboration.
- "Monthly/Weekly/Daily Outlook": narrative & explanatory; connect dots, explain implications without adding new analysis.
- "5min Reviews …": reflective; summarize on the current words and you can add more analysis based on my analysis but only from AL Brooks price action method.

LENGTH TARGETS (to reach ~20–30 minutes total)
- Executive Strip: ~300to 500 words.
- Monthly/Weekly/Daily Outlook: ~600 to 800 words EACH.
- Each 5min Review: ~300 to 500 words 

STRUCTURE & TRANSITIONS
- Use brief transitions to keep flow natural between sections.
- Expand abbreviations on first mention (e.g., “HH (higher high)”) if present in Markdown (usually none).


IMAGE HANDLING
- For each H1 section, extract the FIRST image URL found within that section (if any) and place it in `image_url`.
- If no image is present for a section, set `image_url` to null.
- Do not transform, summarize, or guess image contents; just surface the raw URL string.

SLUG RULE
- Provide a `slug` for each section: lowercase, kebab-case, alphanumeric plus hyphens only.
  - Examples: "Weekly Executive Strip" → "weekly-executive-strip"; "5min Reviews for Last Monday" → "5min-reviews-for-last-monday".

JSON OUTPUT SPEC
- Output a single JSON object with one key: "sections".
- "sections" must be an array, preserving the exact order of H1 sections from the Markdown.
- Each array item is an object with:
  - "title": the exact H1 section title from the Markdown (string)
  - "slug": the normalized slug as specified above (string)
  - "image_url": the first image URL inside that section or null (string or null), do not use markdown format, only url with ""
  - "script": the full spoken script text for that section (string)

- Example shape (illustrative only; do NOT include comments):
{
  "sections": [
    {
      "title": "Weekly Executive Strip",
      "slug": "weekly-executive-strip",
      "image_url": "https://www.tradingview.com/x/XXXXXXX/",
      "script": "…spoken script…"
    },
    {
      "title": "Monthly Outlook",
      "slug": "monthly-outlook",
      "image_url": "https://www.tradingview.com/x/YYYYYYY/",
      "script": "…spoken script…"
    }
  ]
}

STYLE REMINDERS
- Keep it human and confident, but do not invent details.
- If the Markdown includes a date range or session name/front-matter, you may reference it in the Executive Strip’s opening line verbatim.
- Avoid repeating the Executive Strip inside the Weekly Outlook; instead build on it with reasoning supplied by the Markdown.

NOW DO THIS
- Read the Markdown.
- Produce the JSON object as specified, name it as the MD file name after '-' and remove white spaces, e.g. 2025Week37.json, give me json file to download

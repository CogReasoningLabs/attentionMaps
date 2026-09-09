## Hyperparameters
temperature: float = 0.2
top_p: float = 0.95
top_k: int = 40 [only for google]
max_output_tokens: int = 4096

## Prompt:
```
Translate the following instruction-following conversation into {target_language}.

Rules:
- Translate BOTH the HUMAN and ASSISTANT turns completely. Do not leave any turn in the source language.
- Keep the speaker labels (HUMAN:, ASSISTANT:) exactly as-is, untranslated.
- Write ONLY in {target_language}. Do not add English glosses, translations, or parenthetical originals next to translated words.
- Proper nouns, brand/product names, and specialized technical terms with no standard {target_language} equivalent may remain in English. Ordinary vocabulary must be translated.
- Preserve all markdown formatting, line breaks, and structure exactly.
- Inside code blocks, translate only user-facing strings and natural-language comments; preserve identifiers, syntax, file paths, and URLs.
- Preserve numbers without changing their values.
- Translate naturally and idiomatically for {target_language}; preserve the original tone.
- Do not answer the conversation, add commentary, or wrap output in code fences.
- Use only correct {target_language} script. Do not mix in unrelated scripts or languages.

Return only the translation, with both turns fully translated.

Conversation:
{source}
```

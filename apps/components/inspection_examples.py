"""Category definitions and saved, already-classified inspection examples."""


def _observed_writing_system(item):
    parts = [f"{'Other scripts' if name == 'Other' else name} ({item['script_percentages'][name]:.2f}%)"
             for name, count in item["script_counts"].items() if count]
    return ", ".join(parts) if parts else "No eligible letters/marks"


def render_category_definitions(st, status):
    thresholds = status.get("classification_thresholds")
    if not thresholds:
        st.caption("This historical run did not save category thresholds; its original results are shown above.")
        return
    dominant = thresholds["script_dominance_ratio"]
    other = thresholds["script_max_other_ratio"]
    with st.expander("What Nepali-only, Devanagari, Romanized, and Unknown mean"):
        st.markdown(
            "Language and writing system are separate measurements. Devanagari text is not automatically Nepali. "
            "The percentage tables classify individual records; the summary cards show votes over sampling runs."
        )
        st.table([
            {"Language category": "Nepali-only", "Record rule": "Accepted language evidence names only Nepali (ne)."},
            {"Language category": "English-only", "Record rule": "Accepted language evidence names only English (en)."},
            {"Language category": "Bilingual (Nepali-English)", "Record rule": "The record's language labels name both Nepali and English."},
            {"Language category": "Multilingual", "Record rule": "The record has multiple language labels including a language other than Nepali/English."},
            {"Language category": "Other", "Record rule": "Accepted evidence names one language outside Nepali/English."},
            {"Language category": "Unknown", "Record rule": "No accepted language evidence: missing metadata, detection disabled, too short, or a rejected prediction."},
        ])
        st.caption("Evidence order: record language columns, then an explicit Hugging Face language partition, then fastText when enabled. "
                   "fastText predicts one dominant language per record; it does not establish bilingual or multilingual record labels.")
        st.markdown(
            f"For Nepali-eligible records, count Devanagari (D) and Latin (L) letters/marks, and other-script letters (O). "
            f"Spaces, punctuation, digits, and unsupported combining marks do not count. Other-script share must be at most **{other:.0%}**. "
            "Apply these rules in order:"
        )
        st.table([
            {"Script category": "Unknown", "Record rule": "Nepali evidence exists, but there are no eligible letters/marks."},
            {"Script category": "Other", "Record rule": f"O / (D + L + O) exceeds {other:.0%}, or there are no Devanagari/Latin letters."},
            {"Script category": "Devanagari", "Record rule": f"D / (D + L) is at least {dominant:.0%}."},
            {"Script category": "Mixed (Nepali + English)", "Record rule": "Devanagari is below its threshold and language evidence includes English as well as Nepali."},
            {"Script category": "Romanized", "Record rule": f"L / (D + L) is at least {dominant:.0%}, with Nepali evidence and no English label."},
            {"Script category": "Mixed (Devanagari + romanized)", "Record rule": "Both scripts occur; neither reaches the dominance threshold and English is not labelled."},
            {"Script category": "Excluded: no accepted Nepali evidence", "Record rule": "No Nepali-specific category is assigned. The observed writing system is still counted and displayed, even when the language is Unknown."},
        ])
        st.caption("These are Nepali-specific script categories. Romanized requires Nepali evidence. Observed writing systems are shown separately for every example; "
                   "an unknown language does not make its writing system unknown or its text invalid.")
        st.caption(f"Run-level language coverage additionally needs {thresholds['language_min_labelled_ratio']:.0%} labelled/accepted records, "
                   f"retains languages occurring in at least {thresholds['language_min_ratio']:.0%} of labelled records, "
                   f"and requires {thresholds['language_dominance_ratio']:.0%} dominance for a sole retained language. "
                   "Run-level script votes use pooled eligible character counts; they can differ from individual record labels.")
    detection = status.get("language_detection") or {}
    reasons = detection.get("reason_counts", {})
    unknown = status.get("language_no_evidence_records", 0)
    if unknown:
        message = f"Unknown language: {unknown:,} sampled records have no accepted language evidence."
        if detection.get("attempted_records"):
            message += (f" {reasons.get('too_short', 0):,} had fewer than {detection['min_letters']} letters in the detector input; "
                        f"{reasons.get('low_confidence', 0):,} fell below the {detection['min_confidence']:g} confidence cutoff.")
        st.info(message)
        if detection.get("attempted_records"):
            st.caption("The detector's minimum length counts alphabetic letters. Devanagari vowel/combining marks do not count toward that minimum, "
                       "although they can count in the script analysis. A visibly long word can therefore still be too short for language detection.")


def render_record_examples(st, status, *, key):
    render_category_definitions(st, status)
    st.markdown("#### Inspect actual data examples")
    if not status["sampled_records"]:
        st.caption("No records were sampled, so there are no examples to display.")
        return
    saved = status.get("record_examples")
    if saved is None:
        st.info("This report did not save example text. Run a new inspection with percentage sampling to capture examples; earlier results stay unchanged.")
        return
    groups = saved["groups"]
    if not any(groups.values()):
        st.caption("No records were sampled, so there are no examples to display.")
        return
    st.caption(f"Up to {saved['limit_per_category']} random examples per category, selected from the unique records actually analyzed and saved with this run "
               f"(seed {saved['seed']}). Choosing 5 or 10 changes only the display. Categories with fewer records show all available examples.")
    dimension = st.radio("Inspect by", ("Script", "Language"),
                         format_func=lambda value: "Nepali-specific script category" if value == "Script" else value,
                         horizontal=True, key=f"{key}-example-dimension")
    kind = dimension.lower()
    categories = groups[kind]
    if not categories:
        st.caption("No examples were saved for this dimension.")
        return
    category = st.selectbox("Example category", list(categories),
                            format_func=lambda value: "Excluded: no accepted Nepali evidence" if value == "Not applicable" else value,
                            key=f"{key}-example-category-{kind}")
    count = st.radio("Examples to show", (5, 10), horizontal=True, key=f"{key}-example-count")
    examples = categories[category][:count]
    st.caption(f"Showing {len(examples)} saved examples · row reference: {saved['row_reference']}.")
    for index, item in enumerate(examples, 1):
        with st.expander(f"Example {index} · population row {item['population_row']}", expanded=True):
            st.text(item["text"])
            if item["text_truncated"]:
                st.caption(f"Showing the first {len(item['text']):,} of {item['text_characters']:,} characters. Script classification used the full analyzed text.")
            st.write(f"Language: {item['language_category']}")
            st.write(f"Observed writing system: {_observed_writing_system(item)}")
            if item["script_category"] == "Not applicable":
                reason = ("there is no accepted language evidence"
                          if item["language_category"] == "Unknown"
                          else "the accepted language evidence does not include Nepali")
                st.caption(f"Nepali-specific script category: not assigned because {reason}. "
                           "This record is excluded from Nepali-script percentages; its observed writing system is still counted above.")
            else:
                st.write(f"Nepali-specific script category: {item['script_category']}")
            st.caption(f"Language evidence: {item['language_origin']} · accepted codes: {', '.join(item['languages']) or 'none'} "
                       f"· reason: {item['language_reason'].replace('_', ' ')}")
            prediction = item.get("prediction", {})
            input_characters = prediction.get("characters")
            if type(input_characters) is int and input_characters <= len(item["text"]):
                letters = sum(character.isalpha() for character in item["text"][:input_characters])
                minimum = (status.get("language_detection") or {}).get("min_letters")
                st.caption(f"Alphabetic letters in detector input: {letters}" +
                           (f" · required minimum: {minimum}" if minimum is not None else ""))
            if prediction.get("score") is not None:
                st.caption(f"Detector confidence: {prediction['score']:.4f}")
            if prediction.get("truncated"):
                st.caption("Language detector input was truncated at this run's configured character limit.")
            st.table([{"Writing system": name, "Letters/marks": value,
                       "% of all counted letters/marks": item["script_percentages"][name]}
                      for name, value in item["script_counts"].items()])
            deva, latin = item["devanagari_share_of_supported"], item["latin_share_of_supported"]
            if deva is not None:
                st.caption(f"Character shares within Devanagari + Latin: Devanagari {deva:.2%}; Latin {latin:.2%}.")

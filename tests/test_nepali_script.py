import unittest

from attention_maps.explorer.language_status import analyze_language_status, inventory_language_status
from attention_maps.explorer.sampling_vote import RepeatedSampleAnalysis, validate_sampling_result


class NepaliScriptTests(unittest.TestCase):
    def status(self, language, text):
        row = {'text': text}
        if language is not None:
            row['language'] = language
        return analyze_language_status([row], text_columns=['text'])

    def test_nepali_is_required_even_when_devanagari_is_present(self):
        for language in ('en', 'hi', ['en', 'hi'], ['fr', 'de']):
            with self.subTest(language=language):
                status = self.status(language, 'नेपाल English')
                self.assertFalse(status['nepali_covered'])
                self.assertEqual(status['script'], 'Not applicable')
                self.assertEqual(status['script_analysis_counts'], {})
                self.assertIn('Devanagari', status['script_counts'])  # Raw diagnostic stays available.
        unknown = self.status(None, 'नेपाल')
        self.assertIsNone(unknown['nepali_covered'])
        self.assertEqual(unknown['script'], 'Unknown')

    def test_categories_for_nepali_covered_text(self):
        for language, text, expected in (
            ('ne', 'नेपाल सुन्दर छ।', 'Devanagari'),
            ('npi', 'namaste Nepal', 'Romanized'),
            ('nep-Deva', 'नेपाल namaste', 'Mixed (Devanagari + romanized)'),
            (['ne', 'en'], 'नेपाल Hello', 'Mixed (Nepali + English)'),
            ('eng_Latn-npi_Deva', 'namaste Hello', 'Mixed (Nepali + English)'),
            (['ne', 'hi', 'fr'], 'नेपाल', 'Devanagari'),
            (['ne', 'hi'], 'namaste', 'Romanized'),
            (['ne', 'en', 'hi'], 'नेपाल Hello', 'Mixed (Nepali + English)'),
            ('ne', 'नेपाल 中文', 'Unknown'),
            (['ne', 'zh'], 'नेपाल 中文', 'Other'),
            ('ne', '123 !!!', 'Unknown'),
        ):
            with self.subTest(language=language, text=text):
                result = self.status(language, text)
                self.assertTrue(result['nepali_covered'])
                self.assertEqual(result['script'], expected)

    def test_other_language_rows_do_not_change_nepali_script(self):
        status = analyze_language_status([
            {'language': 'ne', 'text': 'namaste'},
            {'language': 'hi', 'text': 'हिन्दी'},
            {'language': 'zh', 'text': '中文'},
            {'text': 'unlabelled नेपाल'},
        ], text_columns=['text'], thresholds={'language_min_labelled_ratio': .75})
        self.assertEqual(status['language_coverage'], 'Multilingual')
        self.assertTrue(status['nepali_covered'])
        self.assertEqual(status['script'], 'Romanized')
        self.assertEqual(status['script_analysis_percentages'], {'Latin': 100})
        self.assertIn('Devanagari', status['script_counts'])
        self.assertIn('Other', status['script_counts'])

    def test_english_nepali_records_can_be_combined_without_unrelated_scripts(self):
        status = analyze_language_status([
            {'language': 'ne', 'text': 'नेपाल'},
            {'language': 'en', 'text': 'Hello'},
            {'language': 'zh', 'text': '中文'},
        ], text_columns=['text'])
        self.assertEqual(status['script'], 'Mixed (Nepali + English)')
        self.assertEqual(set(status['script_analysis_counts']), {'Devanagari', 'Latin'})
        self.assertEqual(set(status['nepali_record_script_counts']), {'Devanagari'})

    def test_hf_selection_qualifies_unlabelled_text_but_not_explicit_other_languages(self):
        inventory = {'format': 'huggingface', 'dataset_config': 'ne', 'dataset_split': 'train'}
        result = inventory_language_status(inventory, [{'text': 'namaste'}])
        self.assertEqual(result['script'], 'Romanized')
        result = inventory_language_status(inventory, [{'text': 'नेपाल', 'language': 'hi'}])
        self.assertEqual(result['script'], 'Not applicable')
        self.assertTrue(result['language_evidence_conflict'])

    def test_all_voting_runs_are_conditional_and_saved_votes_remain_consistent(self):
        settings = {'sample_fraction': .2, 'sampling_runs': 5, 'seed': 42}
        for language, expected, covered in [('ne', 'Devanagari', True), ('hi', 'Not applicable', False), (None, 'Unknown', None)]:
            with self.subTest(language=language):
                analysis = RepeatedSampleAnalysis({}, settings, ['text'], 20)
                for _ in range(20):
                    record = {'text': 'नेपाल'}
                    if language:
                        record['language'] = language
                    analysis.observe(record)
                result = analysis.finish()
                self.assertEqual(result['script'], expected)
                self.assertIs(result['nepali_covered'], covered)
                self.assertEqual(result['voting']['script']['counts'], {expected: 5})
                self.assertTrue(all(run['script'] == expected for run in result['sampling']['run_results']))
                validate_sampling_result(result)

    def test_nepali_only_unknown_abstentions_cannot_vote_other(self):
        settings = {'sample_fraction': 1, 'sampling_runs': 5, 'seed': 42}
        analysis = RepeatedSampleAnalysis({}, settings, ['text'], 3)
        for text in ('नेपाल 中文', 'namaste Привет', '123 !!!'):
            analysis.observe({'text': text, 'language': 'ne'})
        result = analysis.finish()
        self.assertEqual(result['language_coverage'], 'Nepali-only')
        self.assertEqual(result['script'], 'Unknown')
        self.assertEqual(result['voting']['script']['counts'], {'Unknown': 5})
        self.assertEqual(result['voting']['script']['decision'], 'inconclusive')
        self.assertIn('Needs review', result['script_reason'])
        self.assertTrue(all('maximum' in run['script_reason']
                            for run in result['sampling']['run_results']))
        self.assertGreater(result['script_analysis_counts']['Other'], 0)
        validate_sampling_result(result)
        result['script'] = 'Other'
        with self.assertRaisesRegex(ValueError, 'inconsistent with Nepali'):
            validate_sampling_result(result)

    def test_historical_other_vote_remains_readable(self):
        status = self.status('ne', 'नेपाल 中文')
        status.update(script_policy='nepali_required_v1', script='Other')
        validate_sampling_result(status)

    def test_saved_reports_cannot_claim_script_without_nepali_coverage(self):
        status = self.status('hi', 'हिन्दी')
        status['script'] = 'Devanagari'
        with self.assertRaisesRegex(ValueError, 'inconsistent with Nepali'):
            validate_sampling_result(status)

    def test_pooled_nepali_presence_is_not_mistaken_for_majority_presence(self):
        settings = {'sample_fraction': .2, 'sampling_runs': 5, 'seed': 42}
        # Each run chooses one row, with Nepali in a minority of the runs.
        for seed in range(100):
            analysis = RepeatedSampleAnalysis({}, {**settings, 'seed': seed}, ['text'], 5)
            for index in range(5):
                analysis.observe({'language': 'ne' if index == 0 else 'en', 'text': 'नेपाल' if index == 0 else 'Hello'})
            result = analysis.finish()
            if result['pooled_nepali_covered'] and result['nepali_covered'] is False:
                self.assertEqual(result['script'], 'Not applicable')
                validate_sampling_result(result)
                break
        else:
            self.fail('Expected a seed where pooled coverage differs from majority coverage')


if __name__ == '__main__':
    unittest.main()

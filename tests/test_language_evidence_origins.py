import unittest

from attention_maps.explorer.language_status import analyze_language_status, inventory_language_status
from attention_maps.explorer.sampling_vote import RepeatedSampleAnalysis


class LanguageEvidenceOriginsTests(unittest.TestCase):
    def test_manual_card_filter_and_dataset_names_are_not_evidence(self):
        inventories = [
            {'format': 'huggingface', 'dataset_id': 'nepali-corpus', 'dataset_config': 'default', 'dataset_split': 'train', 'declared_languages': ['ne']},
            {'format': 'jsonl', 'provider': 'kaggle', 'dataset_config': 'ne', 'declared_languages': ['ne']},
            {'filter_column': 'language_code', 'filter_value': 'npi'},
            {'row_filters': {'language_code': ['npi']}},
        ]
        for inventory in inventories:
            with self.subTest(inventory=inventory):
                status = inventory_language_status(inventory, [{'text': 'नेपाल'}], declared_languages=['ne'])
                self.assertEqual(status['language_coverage'], 'Unknown')
                self.assertEqual(status['script'], 'Unknown')
                self.assertEqual(status['language_evidence_sources'], [])
        status = analyze_language_status([{'text': 'namaste'}], language_hint='ne', declared_languages=['ne'])
        self.assertEqual(status['language_coverage'], 'Unknown')
        self.assertEqual(status['script'], 'Unknown')

    def test_exact_column_names_and_raw_labels_are_saved(self):
        status = analyze_language_status([
            {'language_code': 'npi', 'text': 'नेपाल'},
            {'language_code': 'nep-Deva', 'text': 'नेपाल'},
            {'source_language': 'en', 'target_language': 'ne', 'text': 'Hello नेपाल'},
        ], text_columns=['text'])
        self.assertEqual(status['language_basis'], 'Dataset column labels')
        self.assertEqual(status['language_coverage'], 'Bilingual (Nepali-English)')
        sources = {source['column']: source for source in status['language_evidence_sources']}
        self.assertEqual(sources['language_code']['raw_label_counts'], {'npi': 1, 'nep-Deva': 1})
        self.assertEqual({source['origin'] for source in sources.values()}, {'dataset_column'})

    def test_only_explicit_hf_partition_labels_supply_fallback(self):
        for field, value, expected in (
            ('dataset_config', 'nep-Deva', 'Nepali-only'),
            ('dataset_config', 'eng_Latn-npi_Deva', 'Bilingual (Nepali-English)'),
            ('dataset_split', 'ne', 'Nepali-only'),
            ('dataset_split', 'train_ne', 'Nepali-only'),
            ('dataset_split', 'en-ne_test', 'Bilingual (Nepali-English)'),
            ('dataset_split', 'train', 'Unknown'),
            ('dataset_split', 'test', 'Unknown'),
            ('dataset_config', 'default', 'Unknown'),
            ('dataset_config', 'news-ne', 'Unknown'),
            ('dataset_config', 'ne-unrelated', 'Unknown'),
        ):
            with self.subTest(field=field, value=value):
                status = inventory_language_status({'format': 'huggingface', field: value}, [{'text': 'नेपाल'}])
                self.assertEqual(status['language_coverage'], expected)
                if expected != 'Unknown':
                    source = status['language_evidence_sources'][0]
                    self.assertEqual(source['origin'], 'huggingface_selection')
                    self.assertEqual(source['value'], value)

    def test_conflicting_sources_are_saved_and_columns_take_priority(self):
        status = inventory_language_status(
            {'format': 'huggingface', 'dataset_config': 'ne', 'dataset_split': 'train'},
            [{'language_code': 'en', 'text': 'English'}])
        self.assertEqual(status['language_coverage'], 'English-only')
        self.assertTrue(status['language_evidence_conflict'])
        self.assertEqual({source['origin'] for source in status['language_evidence_sources']},
                         {'dataset_column', 'huggingface_selection'})

    def test_each_vote_and_unique_pool_preserve_origins(self):
        settings = {'sample_fraction': .2, 'sampling_runs': 5, 'seed': 42, 'languages': ['en']}
        analysis = RepeatedSampleAnalysis({'format': 'huggingface', 'dataset_config': 'ne'}, settings, ['text'], 20)
        for _ in range(20):
            analysis.observe({'language_code': 'npi', 'text': 'नेपाल'})
        result = analysis.finish()
        self.assertEqual(result['voting']['language_coverage']['counts'], {'Nepali-only': 5})
        for run in result['sampling']['run_results']:
            self.assertEqual(run['language_evidence_sources'][0]['raw_label_counts'], {'npi': 4})
        self.assertEqual(result['language_evidence_sources'][0]['raw_label_counts']['npi'], result['sampled_records'])

    def test_missing_evidence_votes_unknown_despite_declarations(self):
        settings = {'sample_fraction': .2, 'sampling_runs': 5, 'seed': 42, 'languages': ['ne']}
        analysis = RepeatedSampleAnalysis({'declared_languages': ['ne']}, settings, ['text'], 20)
        for _ in range(20):
            analysis.observe({'text': 'नेपाल'})
        result = analysis.finish()
        self.assertEqual(result['voting']['language_coverage']['counts'], {'Unknown': 5})
        self.assertEqual(result['voting']['script']['counts'], {'Unknown': 5})


if __name__ == '__main__':
    unittest.main()

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from attention_maps.explorer.classification_thresholds import DEFAULT_THRESHOLDS, resolve_thresholds
from attention_maps.explorer.language_status import analyze_language_status, inventory_language_status
from attention_maps.explorer.inspection_runs import load_inspection_report
from attention_maps.explorer.inspection_history import read_history, load_history_report, decision_text
from attention_maps.explorer.sampling_vote import majority_vote, validate_sampling_result
from scripts.inspect_dataset import main, _arguments


class ClassificationThresholdTests(unittest.TestCase):
    def status(self, text, **thresholds):
        return analyze_language_status([{'language': 'ne', 'text': text}], text_columns=['text'], thresholds=thresholds)

    def test_rare_language_labels_do_not_count_as_bilingual_and_duplicate_columns_do_not_inflate_ratio(self):
        records = ([{'text': 'नेपाल', 'language': 'ne', 'language_code': 'npi'}] * 96 +
                   [{'text': 'English', 'language': 'en'}] * 4)
        status = analyze_language_status(records, text_columns=['text'])
        self.assertEqual(status['language_coverage'], 'Nepali-only')
        self.assertEqual(status['language_record_counts'], {'ne': 96, 'en': 4})
        self.assertEqual(status['language_record_ratios'], {'ne': .96, 'en': .04})
        status = analyze_language_status(records, text_columns=['text'], thresholds={'language_min_ratio': .04})
        self.assertEqual(status['language_coverage'], 'Bilingual (Nepali-English)')
        boundary = analyze_language_status([{'language': 'ne'}]*95 + [{'language': 'en'}]*5)
        self.assertEqual(boundary['language_coverage'], 'Bilingual (Nepali-English)')

    def test_single_language_requires_dominance_not_just_many_small_minorities(self):
        rows = [{'language': 'ne'}]*92 + [{'language': 'en'}]*4 + [{'language': 'hi'}]*4
        status = analyze_language_status(rows)
        self.assertEqual(status['language_coverage'], 'Unknown')
        self.assertIn('single-language coverage requires', status['language_reason'])
        self.assertEqual(analyze_language_status(rows, thresholds={'language_dominance_ratio': .9})['language_coverage'], 'Nepali-only')

    def test_pairs_count_each_language_once_per_record_and_hf_has_no_fabricated_ratios(self):
        status = analyze_language_status([{'language_pair': 'en-ne', 'language': ['ne', 'en'], 'text': 'Hello नेपाल'}])
        self.assertEqual(status['language_record_counts'], {'ne': 1, 'en': 1})
        self.assertEqual(status['language_record_ratios'], {'ne': 1, 'en': 1})
        self.assertEqual(status['language_coverage'], 'Bilingual (Nepali-English)')
        hf = inventory_language_status({'format': 'huggingface', 'dataset_config': 'ne'}, [{'text': 'नेपाल'}])
        self.assertEqual(hf['language_coverage'], 'Nepali-only')
        self.assertEqual(hf['language_record_ratios'], {})
        self.assertEqual(hf['labelled_records'], 0)

    def test_sparse_language_labels_need_sufficient_evidence(self):
        records = [{'text': 'नेपाल', 'language': 'ne'}]*79 + [{'text': 'नेपाल'}]*21
        status = analyze_language_status(records)
        self.assertEqual(status['language_coverage'], 'Unknown')
        self.assertEqual(status['script'], 'Unknown')
        self.assertEqual(status['labelled_record_ratio'], .79)
        self.assertEqual(analyze_language_status(records, thresholds={'language_min_labelled_ratio': .79})['language_coverage'], 'Nepali-only')

    def test_other_script_noise_and_dominance_thresholds_are_independent(self):
        text = 'क'*950 + 'a'*30 + '中'*20
        status = self.status(text)
        self.assertEqual(status['script'], 'Devanagari')
        self.assertEqual(status['script_decision_ratios']['other_share_of_all'], .02)
        self.assertAlmostEqual(status['script_decision_ratios']['devanagari_share_of_supported'], 950/980)
        self.assertEqual(self.status(text, script_dominance_ratio=.99)['script'], 'Mixed (Devanagari + romanized)')
        self.assertEqual(self.status(text, script_max_other_ratio=.01)['script'], 'Unknown')
        self.assertEqual(self.status('क'*95 + '中'*5)['script'], 'Devanagari')
        self.assertEqual(self.status('क'*94 + '中'*6)['script'], 'Unknown')
        self.assertEqual(self.status('क'*95 + 'a'*5)['script'], 'Devanagari')
        self.assertEqual(self.status('क'*94 + 'a'*6)['script'], 'Mixed (Devanagari + romanized)')
        self.assertEqual(self.status('क'*5 + 'a'*95)['script'], 'Romanized')
        self.assertEqual(self.status('中'*10, script_max_other_ratio=1)['script'], 'Unknown')
        self.assertEqual(self.status('123 !')['script'], 'Unknown')

    def test_vote_agreement_keeps_unknown_in_denominator_and_requires_strict_majority(self):
        labels = ['Devanagari']*3 + ['Unknown']*2
        self.assertEqual(majority_vote(labels, .6)['label'], 'Devanagari')
        vote = majority_vote(labels, .8)
        self.assertEqual(vote['label'], 'Unknown')
        self.assertEqual(vote['required_votes'], 4)
        self.assertIn('at least 4', decision_text({'voting': {'script': vote}}, 'script'))
        self.assertEqual(majority_vote(['Devanagari']*3 + ['Romanized']*3, .5)['label'], 'Unknown')
        self.assertEqual(majority_vote(['Devanagari']*7 + ['Unknown']*3, .7)['required_votes'], 7)

    def test_invalid_thresholds_fail_before_source_loading(self):
        for key, value in [('language_min_ratio', 0), ('language_min_ratio', True), ('language_dominance_ratio', .5),
                           ('language_min_labelled_ratio', -1), ('script_dominance_ratio', .5),
                           ('script_max_other_ratio', 1.01), ('vote_min_agreement', .49),
                           ('script_max_other_ratio', float('nan')), ('vote_min_agreement', float('inf'))]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                resolve_thresholds({key: value})
        with self.assertRaisesRegex(ValueError, 'script_dominance_ratio'):
            _arguments(['--local', 'nonexistent.jsonl', '--script-dominance-ratio', '0.2'])

    def test_yaml_cli_legacy_filter_and_saved_ui_share_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, settings, output = root/'source.jsonl', root/'dataset.yaml', root/'history.csv'
            row = {'text': 'क'*97 + 'a'*2 + '中', 'language': 'ne'}
            source.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for _ in range(10)))
            settings.write_text('local: source.jsonl\noutput: history.csv\nscript_dominance_ratio: 0.98\nvote_min_agreement: 0.8\n')
            def run(*flags):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(main(['--settings', str(settings), *flags]), 0, error.getvalue())
                return load_inspection_report(output)
            first = run()
            status = first['language_status']
            self.assertEqual(status['script'], 'Mixed (Devanagari + romanized)')
            self.assertEqual(status['voting']['script']['required_votes'], 4)
            self.assertEqual(first['settings']['script_dominance_ratio'], .98)
            self.assertEqual(status['classification_thresholds'], {**DEFAULT_THRESHOLDS, 'script_dominance_ratio': .98, 'vote_min_agreement': .8})
            self.assertTrue(all(r['classification_thresholds'] == status['classification_thresholds'] for r in status['sampling']['run_results']))
            second = run('--script-dominance-ratio', '.95', '--sample-size', '100')
            self.assertEqual(second['language_status']['script'], 'Devanagari')
            self.assertEqual(load_history_report(output, read_history(output)[0]), first)
            kept = root/'kept.jsonl'
            third = run('--script-max-other-ratio', '0', '--min-devanagari-ratio', '.8',
                        '--filtered-output', str(kept))
            self.assertEqual(third['filter_result']['rows_kept'], 10)
            self.assertEqual(third['filter_result']['language_status']['script'], 'Unknown')
            self.assertEqual(read_history(output)[-1]['Filtered Script'], '')
            self.assertEqual(len(read_history(output)), 3)
            source.unlink()
            from streamlit.testing.v1 import AppTest
            app = AppTest.from_string('''
import streamlit as st
from pathlib import Path
from attention_maps.explorer.inspection_runs import load_inspection_report
from apps.components.inspection_reports import render_inspection_report
render_inspection_report(st, load_inspection_report(Path(PATH)))
'''.replace('PATH', repr(str(output))), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertTrue(any(x.label == 'Classification thresholds and measured ratios' for x in app.expander))
            status['classification_thresholds']['vote_min_agreement'] = 1
            with self.assertRaises(ValueError):
                validate_sampling_result(status)


if __name__ == '__main__':
    unittest.main()

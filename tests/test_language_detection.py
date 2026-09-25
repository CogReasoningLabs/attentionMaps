import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from attention_maps.explorer.language_detection import FastTextLanguageDetector, detection_settings, make_language_detector
from attention_maps.explorer.language_status import analyze_language_status, inventory_language_status
from attention_maps.explorer.sampling_vote import RepeatedSampleAnalysis, validate_sampling_result
from attention_maps.explorer.inspection_runs import load_inspection_report
from scripts.inspect_dataset import _arguments, main


class StubDetector:
    def __init__(self):
        self.calls = []
        self.metadata = {'method': 'fasttext', 'model_path': '/fake/model.ftz', 'model_sha256': 'fixture',
                         'min_confidence': .8, 'min_letters': 20, 'max_characters': 4000}

    def detect(self, text):
        self.calls.append(text)
        language = 'en' if text.startswith('English') else None if text.startswith('?') else 'ne'
        return {'language': language, 'score': .99 if language else .4,
                'truncated': False, 'reason': 'accepted' if language else 'low_confidence'}


class LanguageDetectionTests(unittest.TestCase):
    def test_detector_validates_config_before_loading(self):
        self.assertIsNone(make_language_detector())
        self.assertEqual(Path(detection_settings()['language_detection_model']).name, 'lid.176.bin')
        for settings in ({'language_detection': 'bad'}, {'language_detection_min_confidence': float('nan')},
                         {'language_detection_min_confidence': True}, {'language_detection_min_letters': 0},
                         {'language_detection_max_characters': 5}, {'language_detection_model': 'bad.txt'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                detection_settings(settings)

    def test_batch_api_threshold_short_text_and_truncation(self):
        detector = FastTextLanguageDetector({'language_detection': 'fasttext', 'language_detection_min_letters': 2,
                                            'language_detection_max_characters': 10})
        model = Mock()
        model.predict.return_value = ([['__label__ne']], [[.8]])
        detector._model = model
        self.assertEqual(detector.detect('1 !')['reason'], 'too_short')
        model.predict.assert_not_called()
        result = detector.detect('ab\nc\x00defghijklmnop')
        self.assertEqual(result['language'], 'ne')
        self.assertTrue(result['truncated'])
        self.assertEqual(result['characters'], 10)
        model.predict.assert_called_once_with(['ab c defgh'], k=1)
        model.predict.return_value = ([['__label__ne']], [[.79999]])
        result = detector.detect('abcd')
        self.assertIsNone(result['language'])
        self.assertEqual(result['reason'], 'low_confidence')
        model.predict.return_value = ([['__label__ne']], [[float('nan')]])
        self.assertEqual(detector.detect('abcd')['reason'], 'invalid_prediction')

    def test_missing_custom_model_fails_instead_of_fabricating_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = FastTextLanguageDetector({'language_detection_model': str(Path(directory)/'custom.bin')})
            with patch.dict('sys.modules', {'fasttext': Mock()}), self.assertRaisesRegex(ValueError, 'does not exist'):
                detector.detect('A long enough sentence to invoke language identification.')

    def test_metadata_precedes_detector_for_each_provider(self):
        for inventory in ({'provider': 'kaggle'}, {'provider': 'local'}, {'format': 'huggingface'}):
            detector = StubDetector()
            status = inventory_language_status(inventory, [{'language': 'en', 'text': 'नेपाल'}], detector=detector)
            self.assertEqual(status['language_coverage'], 'English-only')
            self.assertEqual(detector.calls, [])
        detector = StubDetector()
        status = inventory_language_status({'format': 'huggingface', 'dataset_config': 'ne'},
                                           [{'text': 'नेपाल'}], detector=detector)
        self.assertEqual(status['language_coverage'], 'Nepali-only')
        self.assertEqual(detector.calls, [])

    def test_metadata_absent_predictions_supply_separate_provenance_on_all_sources(self):
        for inventory in ({'provider': 'kaggle'}, {'provider': 'local'},
                          {'format': 'huggingface', 'dataset_config': 'default', 'dataset_split': 'train'}):
            detector = StubDetector()
            record = {'text': 'नेपाल'}
            status = inventory_language_status(inventory, [record], detector=detector)
            self.assertEqual(status['language_coverage'], 'Nepali-only')
            self.assertEqual(status['script'], 'Devanagari')
            self.assertEqual(status['observed_language_counts'], {})
            self.assertEqual(status['language_evidence_sources'][0]['origin'], 'text_language_detector')
            self.assertEqual(status['language_detection']['model_sha256'], 'fixture')
            self.assertEqual(record, {'text': 'नेपाल'})

    def test_rejections_remain_in_coverage_denominator_and_predictions_use_same_thresholds(self):
        detector = StubDetector()
        records = [{'text': 'नेपाल'}]*8 + [{'text': '?uncertain'}]*2
        status = analyze_language_status(records, detector=detector)
        self.assertEqual(status['labelled_record_ratio'], .8)
        self.assertEqual(status['language_coverage'], 'Nepali-only')
        self.assertEqual(status['language_detection']['rejected_records'], 2)
        self.assertEqual(analyze_language_status(records+[{'text': '?uncertain'}], detector=detector)['language_coverage'], 'Unknown')
        records = [{'text': 'नेपाल'}]*96 + [{'text': 'English'}]*4
        self.assertEqual(analyze_language_status(records, detector=detector)['language_coverage'], 'Nepali-only')
        self.assertEqual(analyze_language_status(records, detector=detector, thresholds={'language_min_ratio': .04})['language_coverage'], 'Bilingual (Nepali-English)')
        records = [{'text': 'नेपाल', 'language': 'ne'}]*8 + [{'text': 'English'}]*2
        status = analyze_language_status(records, detector=detector)
        self.assertEqual(status['language_coverage'], 'Bilingual (Nepali-English)')
        self.assertEqual(status['language_detection']['attempted_records'], 2)
        self.assertEqual({s['origin'] for s in status['language_evidence_sources']}, {'dataset_column', 'text_language_detector'})

    def test_only_unique_sampled_positions_are_detected_and_votes_preserve_unknown(self):
        detector = StubDetector()
        settings = {'seed': 42, 'sample_fraction': .2, 'sampling_runs': 5}
        with patch('attention_maps.explorer.sampling_vote.make_language_detector', return_value=detector):
            analysis = RepeatedSampleAnalysis({}, settings, ['text'], 100)
            for index in range(100):
                analysis.observe({'text': f'नेपाल {index}'})
        status = analysis.finish()
        validate_sampling_result(status)
        self.assertEqual(len(detector.calls), status['sampling']['unique_sampled_records'])
        self.assertLess(len(detector.calls), 100)
        self.assertEqual(len(set(detector.calls)), len(detector.calls))
        self.assertEqual(status['voting']['language_coverage']['counts'], {'Nepali-only': 5})
        for run in status['sampling']['run_results']:
            self.assertEqual(run['language_detection']['accepted_records'], 20)
        with patch('attention_maps.explorer.sampling_vote.make_language_detector', return_value=StubDetector()):
            analysis = RepeatedSampleAnalysis({}, settings, ['text'], 100)
            for _ in range(100):
                analysis.observe({'text': '?uncertain'})
        status = analysis.finish()
        self.assertEqual(status['voting']['language_coverage']['counts'], {'Unknown': 5})
        self.assertEqual(status['script'], 'Unknown')

    def test_prediction_prefix_does_not_truncate_script_analysis(self):
        detector = FastTextLanguageDetector({'language_detection_max_characters': 20})
        detector._model = Mock()
        detector._model.predict.return_value = ([['__label__ne']], [[.99]])
        status = analyze_language_status([{'text': 'क'*20 + 'a'*980}], detector=detector)
        self.assertEqual(status['script'], 'Romanized')
        self.assertEqual(status['sampled_characters'], 1000)
        self.assertEqual(status['language_detection']['truncated_records'], 1)

    def test_cli_yaml_selection_legacy_filter_csv_and_read_only_ui(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, settings, output = root/'source.jsonl', root/'dataset.yaml', root/'history.csv'
            source.write_text(''.join(json.dumps({'text': 'नेपाल', 'part': part}, ensure_ascii=False)+'\n'
                                      for part in ['keep']*10+['skip']*10), encoding='utf-8')
            settings.write_text('local: source.jsonl\noutput: history.csv\ntext_columns: [text]\n'
                                'row_filters: {part: [keep]}\nlanguage_detection: fasttext\n'
                                'language_detection_model: models/custom.ftz\n', encoding='utf-8')
            _, parsed = _arguments(['--settings', str(settings), '--language-detection-min-confidence', '.9'])
            self.assertEqual(parsed['language_detection_model'], str(root/'models/custom.ftz'))
            self.assertEqual(parsed['language_detection_min_confidence'], .9)
            def run(*flags):
                detector = StubDetector()
                with patch('attention_maps.explorer.sampling_vote.make_language_detector', return_value=detector), \
                     patch('attention_maps.explorer.inspection_runs.make_language_detector', return_value=detector), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(main(['--settings', str(settings), *flags]), 0, error.getvalue())
                return load_inspection_report(output), detector
            report, detector = run()
            self.assertEqual(report['inventory']['row_selection']['rows_matched'], 10)
            self.assertEqual(len(detector.calls), report['language_status']['sampling']['unique_sampled_records'])
            report, _ = run('--sample-size', '100')
            self.assertEqual(report['language_status']['language_detection']['accepted_records'], 10)
            report, _ = run('--min-devanagari-ratio', '.8', '--filtered-output', str(root/'kept.jsonl'))
            self.assertEqual(report['filter_result']['language_status']['language_coverage'], 'Nepali-only')
            source.unlink()
            from streamlit.testing.v1 import AppTest
            with patch.object(FastTextLanguageDetector, '_load', side_effect=AssertionError('UI must not load model')):
                app = AppTest.from_string('''
import streamlit as st
from pathlib import Path
from attention_maps.explorer.inspection_runs import load_inspection_report
from apps.components.inspection_reports import render_inspection_report
render_inspection_report(st, load_inspection_report(Path(PATH)))
'''.replace('PATH', repr(str(output))), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertTrue(any(x.label == 'Text language detector and rejected predictions' for x in app.expander))


if __name__ == '__main__':
    unittest.main()

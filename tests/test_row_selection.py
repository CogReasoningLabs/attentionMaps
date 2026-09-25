import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.explorer.inspection_runs import build_inspection_report, iter_inspection_records, load_inspection_report
from attention_maps.explorer.inspection_history import read_history, load_history_report, PRE_SELECTION_FIELDS
from attention_maps.explorer.language_status import analyze_language_status, language_context
from attention_maps.explorer.row_selection import matches_row_filters, validate_row_filters
from attention_maps.explorer.semantic_source import source_settings, resolve_source
from apps.components.inspection_reports import report_matches_source
from scripts.inspect_dataset import _arguments, main
from scripts.cluster_dataset import parser


class RowSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / 'mixed.jsonl'
        self.output = self.root / 'history.csv'
        # Matching records occur late in the file: selection must not sample a prefix first.
        self.records = ([{'text': 'English', 'language_code': 'en'} for _ in range(80)] +
                        [{'text': 'नेपाल', 'language_code': 'npi'} for _ in range(20)])
        self.write_records(self.records)

    def write_records(self, records):
        self.source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))

    def run_inspect(self, *flags):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
            result = main(['--local', str(self.source), '--output', str(self.output), *flags])
        self.assertEqual(result, 0, error.getvalue())
        return load_inspection_report(self.output)

    def test_filter_before_percentage_sampling_csv_and_read_only_ui(self):
        report = self.run_inspect('--row-filter', 'language_code=npi')
        status = report['language_status']
        self.assertEqual(status['language_coverage'], 'Nepali-only')
        self.assertEqual(status['script'], 'Devanagari')
        self.assertEqual(status['sampling']['population_rows'], 20)
        self.assertEqual(status['sampling']['rows_per_run'], 4)
        self.assertEqual([run['sampled_records'] for run in status['sampling']['run_results']], [4]*5)
        self.assertEqual(status['voting']['language_coverage']['counts'], {'Nepali-only': 5})
        self.assertEqual(report['inventory']['row_selection']['rows_excluded'], 80)
        self.assertEqual(report['sample_scope'], 'language_selected_sample')
        row = read_history(self.output)[0]
        self.assertEqual((row['Rows before language selection'], row['Rows after language selection']), ('100', '20'))
        self.assertEqual(load_history_report(self.output, row), report)
        self.source.unlink()
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
from pathlib import Path
import streamlit as st
from attention_maps.explorer.inspection_runs import load_inspection_report
from apps.components.inspection_reports import render_inspection_report
render_inspection_report(st, load_inspection_report(Path(PATH)))
'''.replace('PATH', repr(str(self.output))), default_timeout=20)
        with patch('attention_maps.explorer.inspection_runs.iter_inspection_records', side_effect=AssertionError('No source reads')):
            app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any('20 matching rows out of 100' in x.value for x in app.markdown))
        self.assertEqual(next(x.value for x in app.metric if x.label == 'Language coverage'), 'Nepali-only')

    def test_legacy_and_devanagari_filter_only_consume_language_selection(self):
        report = self.run_inspect('--row-filter', 'language_code=npi', '--sample-size', '100')
        self.assertEqual(report['language_status']['sampled_records'], 20)
        self.assertEqual(report['language_status']['script'], 'Devanagari')
        kept = self.root / 'kept.jsonl'
        report = self.run_inspect('--row-filter', 'language_code=npi', '--min-devanagari-ratio', '1',
                                  '--max-records', '10', '--filtered-output', str(kept))
        self.assertEqual(report['inventory']['rows'], 20)
        self.assertEqual(report['language_status']['sampling']['population_rows'], 10)
        self.assertEqual(report['language_status']['sampling']['rows_per_run'], 2)
        self.assertEqual(report['filter_result']['rows_kept'], 10)
        self.assertTrue(all(json.loads(line)['language_code'] == 'npi' for line in kept.read_text().splitlines()))

    def test_pairs_and_script_qualified_labels_are_exact_selectors(self):
        self.write_records([{'text': 'Hello नेपाल', 'language_pair': 'en-ne'}]*10 +
                           [{'text': 'Other', 'language_pair': 'en-fr'}]*90)
        report = self.run_inspect('--row-filter', 'language_pair=en-ne')
        self.assertEqual(report['settings']['text_columns'], ['text'])
        self.assertEqual(report['language_status']['language_coverage'], 'Bilingual (Nepali-English)')
        self.assertEqual(report['language_status']['script'], 'Mixed (Nepali + English)')
        self.assertEqual(report['language_status']['sampling']['rows_per_run'], 2)
        self.write_records([{'text': 'नेपाल', 'language_code': code} for code in ('nep-Deva', 'ne', 'en', 'npi')]*10)
        report = self.run_inspect('--row-filter', 'language_code=nep-Deva', '--row-filter', 'language_code=ne')
        self.assertEqual(report['inventory']['rows'], 20)
        self.assertEqual(report['language_status']['language_coverage'], 'Nepali-only')
        self.assertEqual(language_context({'format': 'huggingface', 'dataset_config': 'en-ne'})['hf_selection'][0]['value'], 'en-ne')
        status = analyze_language_status([{'text': 'Hello नेपाल'}], text_columns=['text'],
                                         **language_context({'format': 'huggingface', 'dataset_config': 'en-ne'}))
        self.assertEqual(status['language_coverage'], 'Bilingual (Nepali-English)')

    def test_two_columns_nested_labels_lists_and_case_sensitive_matching(self):
        filters = {'source_language': ['en'], 'target_language': ['ne', 'npi']}
        self.assertTrue(matches_row_filters({'source_language': 'en', 'target_language': 'npi'}, filters))
        self.assertFalse(matches_row_filters({'source_language': 'fr', 'target_language': 'ne'}, filters))
        self.assertFalse(matches_row_filters({'source_language': 'en', 'target_language': 'NE'}, filters))
        self.assertFalse(matches_row_filters({'source_language': 'en'}, filters))
        self.assertTrue(matches_row_filters({'meta': {'language': ['ne', 'en']}}, {'meta.language': ['ne']}))
        self.assertFalse(matches_row_filters({'language': 'nep-Deva'}, {'language': ['ne']}))
        # Membership retains whole bilingual records, not only the matching language portion.
        status = analyze_language_status([{'language': ['ne', 'en'], 'text': 'Hello नेपाल'}], text_columns=['text'])
        self.assertEqual(status['language_coverage'], 'Bilingual (Nepali-English)')
        self.write_records([{'text': 'Hello नेपाल', 'source_language': 'en', 'target_language': 'ne'}]*10 +
                           [{'text': 'Other', 'source_language': 'fr', 'target_language': 'ne'}]*10)
        report = self.run_inspect('--row-filter', 'source_language=en', '--row-filter', 'target_language=ne')
        self.assertEqual(report['inventory']['rows'], 10)
        self.assertEqual(report['language_status']['language_coverage'], 'Bilingual (Nepali-English)')
        self.assertEqual(report['settings']['text_columns'], ['text'])

    def test_bad_selection_never_publishes_and_reports_observed_labels(self):
        for value in ([], {'language': []}, {'language': 'ne'}, {'': ['ne']}, {'language': [1]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_row_filters(value)
        for flags, message in ((['--row-filter', 'language_code=not-a-label'], 'Observed label examples'),
                               (['--row-filter', 'missing=ne'], 'Unknown row-filter'),
                               (['--row-filter', 'language_code=npi', '--formats-only'], 'formats_only')):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(['--local', str(self.source), '--output', str(self.output), *flags]), 1)
            self.assertIn(message, error.getvalue())
            self.assertFalse(self.output.exists())

    def test_yaml_cli_overrides_and_semantic_readers_share_filters(self):
        config = self.root / 'dataset.yaml'
        config.write_text('local: mixed.jsonl\nrow_filters:\n  language_code: [npi]\n')
        flags = ['--settings', str(config)]
        self.assertEqual(_arguments(flags)[1]['row_filters'], {'language_code': ['npi']})
        self.assertEqual(_arguments([*flags, '--no-row-filters'])[1]['row_filters'], {})
        self.assertEqual(_arguments([*flags, '--row-filter', 'language_code=en'])[1]['row_filters'], {'language_code': ['en']})
        with self.assertRaises(ValueError):
            _arguments([*flags, '--row-filter', 'language_code=en', '--no-row-filters'])
        args = parser().parse_args(['run', '--source-settings', str(config), '--output-dir', str(self.root / 'embeddings')])
        inventory, fields = resolve_source(source_settings(args))
        self.assertEqual(fields, ['text'])
        self.assertIsNone(inventory['rows'])  # Full source count cannot describe the filtered subset.
        self.assertEqual(len(list(iter_inspection_records(inventory))), 20)
        report = self.run_inspect('--row-filter', 'language_code=npi')
        self.assertFalse(report_matches_source(report, {**report['inventory'], 'row_filters': {}}, SimpleNamespace(files=(self.source,))))

    def test_huggingface_counts_matching_rows_in_pinned_selected_shards(self):
        inventory = {'format': 'huggingface', 'dataset_id': 'owner/multi', 'dataset_config': 'default',
                     'dataset_revision': 'pinned', 'dataset_split': 'train', 'dataset_shards': ['later.jsonl'],
                     'rows': 100, 'columns': ['text', 'language_code'], 'schema': [], 'declared_languages': ['en', 'ne', 'fr']}
        settings = {'sample_fraction': 0.2, 'sampling_runs': 5, 'sampling_method': 'random', 'seed': 42,
                    'text_columns': ['text'], 'row_filters': {'language_code': ['npi']}}
        with patch('datasets.load_dataset', side_effect=lambda *a, **kw: iter(self.records)) as loader, contextlib.redirect_stderr(io.StringIO()):
            report = build_inspection_report(inventory, settings)
        self.assertEqual(loader.call_count, 2)
        for call in loader.call_args_list:
            self.assertEqual(call.kwargs['revision'], 'pinned')
            self.assertEqual(call.kwargs['data_files'], {'train': ['later.jsonl']})
        self.assertEqual(report['language_status']['sampling']['population_rows'], 20)
        self.assertEqual(report['language_status']['language_coverage'], 'Nepali-only')
        self.assertEqual(inventory['rows'], 100)  # Caller inventory remains unchanged.

    def test_hf_portion_precedes_sampling_thresholds_and_devanagari_export(self):
        # Excluded records also pass the script threshold: metadata selection must
        # happen first, not after sampling or after the script export.
        outside = [{'text': 'नेपाल', 'language_code': 'en', 'id': i} for i in range(100)]
        selected = [{'text': 'नेपाल' if i % 2 else 'namaste', 'language_code': 'npi', 'id': 100+i}
                    for i in range(20)]
        inventory = {'format': 'huggingface', 'dataset_id': 'owner/multilingual',
                     'dataset_config': 'chosen-config', 'dataset_split': 'validation',
                     'dataset_revision': 'pinned', 'dataset_shards': ['chosen-shard.jsonl'],
                     'rows': 120, 'columns': ['text', 'language_code', 'id'], 'schema': [],
                     'declared_languages': ['en', 'ne']}
        settings = {'sample_fraction': .2, 'sampling_runs': 5, 'sampling_method': 'random',
                    'seed': 42, 'text_columns': ['text'], 'row_filters': {'language_code': ['npi']},
                    'min_devanagari_ratio': .8, 'filtered_output': str(self.root / 'kept.jsonl')}
        with patch('datasets.load_dataset', side_effect=lambda *a, **kw: iter(outside + selected)) as loader, \
             contextlib.redirect_stderr(io.StringIO()):
            report = build_inspection_report(inventory, settings)
        status = report['language_status']
        self.assertEqual(report['inventory']['row_selection']['rows_scanned'], 120)
        self.assertEqual(report['inventory']['rows'], 20)
        self.assertEqual(status['sampling']['population_rows'], 20)
        self.assertEqual(status['sampling']['rows_per_run'], 4)
        self.assertTrue(all(run['language_record_ratios'] == {'ne': 1.0}
                            and run['labelled_record_ratio'] == 1
                            for run in status['sampling']['run_results']))
        exported = [json.loads(line) for line in Path(settings['filtered_output']).read_text().splitlines()]
        self.assertEqual({row['id'] for row in exported}, {row['id'] for row in selected if row['text'] == 'नेपाल'})
        self.assertEqual(report['filter_result']['rows_scanned'], 20)
        self.assertEqual(report['filter_result']['rows_kept'], 10)
        self.assertEqual(report['filter_result']['language_status']['sampling']['population_rows'], 10)
        self.assertEqual(report['filter_result']['language_status']['sampling']['rows_per_run'], 2)
        self.assertEqual(loader.call_count, 2)
        for call in loader.call_args_list:
            self.assertEqual(call.args, ('owner/multilingual', 'chosen-config'))
            self.assertEqual(call.kwargs['split'], 'validation')
            self.assertEqual(call.kwargs['revision'], 'pinned')
            self.assertEqual(call.kwargs['data_files'], {'validation': ['chosen-shard.jsonl']})

    def test_older_csv_is_upgraded_on_append_without_losing_old_report(self):
        old_report = self.run_inspect()
        rows = read_history(self.output)
        with self.output.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=PRE_SELECTION_FIELDS)
            writer.writeheader()
            writer.writerows({key: row[key] for key in PRE_SELECTION_FIELDS} for row in rows)
        self.run_inspect('--row-filter', 'language_code=npi')
        rows = read_history(self.output)
        self.assertEqual(len(rows), 2)
        self.assertEqual(load_history_report(self.output, rows[0]), old_report)
        self.assertEqual(rows[1]['Rows after language selection'], '20')


if __name__ == '__main__':
    unittest.main()

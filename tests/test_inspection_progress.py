import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.inspect_dataset import main


class Terminal(io.StringIO):
    def isatty(self):
        return True


class InspectionProgressTests(unittest.TestCase):
    def test_jsonl_selection_and_sampling_show_separate_bars_without_corrupting_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.jsonl'
            source.write_text(''.join(json.dumps({'text': 'नेपाल', 'language_code': 'npi' if i % 2 else 'eng'}, ensure_ascii=False) + '\n'
                                      for i in range(12)), encoding='utf-8')
            output, terminal = io.StringIO(), Terminal()
            with contextlib.redirect_stdout(output), patch('sys.stderr', terminal):
                code = main(['--local', str(source), '--row-filter', 'language_code=npi',
                             '--sample-fraction', '.5', '--sampling-runs', '3',
                             '--output', str(root/'report.csv')])
            self.assertEqual(code, 0, terminal.getvalue())
            report = json.loads(output.getvalue())
            self.assertEqual(report['inventory']['row_selection']['rows_matched'], 6)
            self.assertEqual(report['language_status']['sampling']['rows_per_run'], 3)
            progress = terminal.getvalue()
            self.assertIn('Read JSONL', progress)
            self.assertIn('Match source rows', progress)
            self.assertIn('Sample and classify', progress)
            self.assertIn('100%', progress)

    def test_txt_inventory_and_filtered_output_show_progress_without_changing_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.txt'
            source.write_text('नेपाल\nEnglish\nनेपाली भाषा\nabc\nनेपाल\n', encoding='utf-8')
            output, terminal = io.StringIO(), Terminal()
            with contextlib.redirect_stdout(output), patch('sys.stderr', terminal):
                code = main(['--local', str(source), '--sample-fraction', '.5', '--sampling-runs', '3',
                             '--min-devanagari-ratio', '.5', '--filtered-output', str(root/'kept.jsonl'),
                             '--output', str(root/'report.csv')])
            self.assertEqual(code, 0, terminal.getvalue())
            report = json.loads(output.getvalue())
            self.assertEqual(report['filter_result']['rows_kept'], 3)
            self.assertEqual(report['filter_result']['rows_scanned'], 5)
            progress = terminal.getvalue()
            self.assertIn('Read TXT', progress)
            self.assertIn('Filter selected rows', progress)
            self.assertIn('Sample and classify', progress)
            self.assertIn('100%', progress)


if __name__ == '__main__':
    unittest.main()

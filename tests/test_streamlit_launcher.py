import signal
import unittest
from unittest.mock import Mock, patch

from scripts.run_dataset_explorer import (
    ShutdownRequest,
    _normal_exit_code,
    _signal_process_tree,
    streamlit_command,
)


class StreamlitLauncherTests(unittest.TestCase):
    def test_builds_dataset_explorer_command_and_forwards_options(self):
        command = streamlit_command(["--", "--server.port=8502"])

        self.assertEqual(command[1:4], ["-m", "streamlit", "run"])
        self.assertTrue(command[4].endswith("apps/dataset_explorer.py"))
        self.assertEqual(command[5:], ["--server.port=8502"])

    def test_second_signal_requests_force_shutdown(self):
        request = ShutdownRequest()

        request.handle(signal.SIGINT, None)
        self.assertTrue(request.requested)
        self.assertFalse(request.force_requested)
        request.handle(signal.SIGINT, None)
        self.assertTrue(request.force_requested)

    @patch("scripts.run_dataset_explorer.os.killpg")
    def test_signals_complete_posix_process_group(self, killpg):
        process = Mock(pid=321)
        process.poll.return_value = None

        with patch("scripts.run_dataset_explorer.os.name", "posix"):
            _signal_process_tree(process, signal.SIGTERM)

        killpg.assert_called_once_with(321, signal.SIGTERM)

    def test_converts_signal_return_code_to_shell_exit_code(self):
        self.assertEqual(_normal_exit_code(-signal.SIGTERM), 143)
        self.assertEqual(_normal_exit_code(0), 0)


if __name__ == "__main__":
    unittest.main()

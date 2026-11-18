"""A completed long batch must not overwrite an earlier checkpoint's pass."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks import model_census


class CensusCompletionTest(unittest.TestCase):
    def test_late_controller_poll_keeps_completed_reports(self):
        with tempfile.TemporaryDirectory(prefix='aircolibri-census-') as directory:
            output = Path(directory)
            entries = [{'model':'first'}, {'model':'last'}]

            class CompletedWorker:
                returncode = 0

                def __init__(self, *args, **kwargs):
                    self.polls = 0
                    # The worker completes after the controller starts, but before
                    # its next poll; the *batch* has exceeded the per-model budget.
                    for entry in entries:
                        model_census.write(output/(entry['model']+'.json'),
                            dict(**entry,status='passed',stage='complete'))
                    model_census.write(output/'worker_state.json',dict(phase='complete'))

                def poll(self):
                    self.polls += 1
                    return None if self.polls == 1 else 0

            args = SimpleNamespace(output=output,engine_root=output,models=output,
                weight_format='q6',tokens=2,prefill_chunk=8,continuation_tokens=1,
                prompt='Hi',timeout=600,pause_file=None)
            with patch.object(model_census.subprocess,'Popen',CompletedWorker), \
                 patch.object(model_census.time,'monotonic',side_effect=[0,700]), \
                 patch.object(model_census.time,'sleep'), \
                 patch.object(model_census,'kill_worker',side_effect=AssertionError('completed batch killed')):
                self.assertEqual(model_census.persistent_census(args,entries),0)
            reports = [json.loads((output/(e['model']+'.json')).read_text()) for e in entries]
            self.assertTrue(all(r['status']=='passed' for r in reports))
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual([r['status'] for r in summary],['passed','passed'])

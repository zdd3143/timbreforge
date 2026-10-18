"""Persistent job lifecycle and the serial heavy-task resource manager."""
from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from timbreforge.job_queue import JobQueue, JobQueueError
from timbreforge.resource_manager import (ResourceManager, available_memory_bytes,
                                          clear_cleanup_callbacks, ensure_memory,
                                          register_cleanup, release_low_priority_resources)


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = JobQueue(Path(self.temp.name) / 'jobs')

    def test_create_persist_and_transition(self):
        job = self.queue.create('generation', role_id='a' * 32,
                                payload={'role_id': 'a' * 32}, priority=3)
        self.assertEqual(job['state'], 'waiting')
        reloaded = JobQueue(self.queue.root).get(job['id'])
        self.assertEqual(reloaded['payload'], {'role_id': 'a' * 32})
        self.assertEqual(reloaded['priority'], 3)
        self.queue.transition(job['id'], 'queued', message='排队')
        preparing = self.queue.transition(job['id'], 'preparing', message='准备模型')
        self.assertIn('准备', preparing['message'])
        self.assertIsNotNone(preparing['started_at'])
        self.queue.transition(job['id'], 'generating')
        self.queue.transition(job['id'], 'saving')
        self.queue.transition(job['id'], 'cleaning')
        done = self.queue.transition(job['id'], 'completed', result={'file': 'a.wav'})
        self.assertEqual(done['state'], 'completed')
        self.assertIsNotNone(done['finished_at'])
        self.assertEqual(done['result'], {'file': 'a.wav'})

    def test_invalid_transitions_and_payloads(self):
        job = self.queue.create('generation')
        with self.assertRaises(JobQueueError):
            self.queue.transition(job['id'], 'completed')
        with self.assertRaises(JobQueueError):
            self.queue.create('generation', payload={'bad': float('nan')})
        with self.assertRaises(JobQueueError):
            self.queue.create('')
        with self.assertRaises(JobQueueError):
            self.queue.create('x', priority=99)
        with self.assertRaises(JobQueueError):
            self.queue.get('missing')

    def test_cancel_request_is_visible(self):
        job = self.queue.create('generation')
        self.assertFalse(self.queue.cancel_requested(job['id']))
        requested = self.queue.request_cancel(job['id'])
        self.assertTrue(requested['cancel_requested'])
        self.queue.transition(job['id'], 'cancelled')
        self.assertTrue(self.queue.cancel_requested(job['id']))

    def test_recover_interrupted_marks_active_jobs_failed(self):
        first = self.queue.create('generation')
        second = self.queue.create('generation')
        self.queue.transition(first['id'], 'queued')
        self.queue.transition(first['id'], 'preparing')
        self.queue.transition(second['id'], 'cancelled')
        recovered = self.queue.recover_interrupted()
        self.assertEqual(recovered, [first['id']])
        job = self.queue.get(first['id'])
        self.assertEqual(job['state'], 'failed')
        self.assertIn('重新生成', job['error'])
        self.assertEqual(self.queue.get(second['id'])['state'], 'cancelled')

    def test_list_is_newest_first_and_bounded(self):
        ids = [self.queue.create('generation')['id'] for _ in range(3)]
        listed = self.queue.list(limit=2)
        self.assertEqual(len(listed), 2)
        self.assertEqual(listed[0]['id'], ids[-1])


class ResourceManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = JobQueue(self.root / 'jobs')
        self.release_patch = patch('timbreforge.resource_manager.release_idle_models',
                                   return_value='stub')
        self.release_patch.start()
        self.addCleanup(self.release_patch.stop)
        clear_cleanup_callbacks()
        self.addCleanup(clear_cleanup_callbacks)
        self.manager = ResourceManager(jobs=self.jobs,
                                       workspace_root=self.root / 'workspaces')
        self.addCleanup(self.manager.shutdown)

    def test_tasks_run_serially_and_workspace_is_cleaned(self):
        order = []

        def task(name):
            def run(context):
                order.append(('start', name, context.workspace.is_dir()))
                time.sleep(.05)
                order.append(('end', name))
                return {'name': name}
            return run

        first = self.manager.submit('generation', task('a'))
        second = self.manager.submit('generation', task('b'))
        for job in (first, second):
            finished = self.manager.wait(job['id'], timeout=5)
            self.assertEqual(finished['state'], 'completed')
            self.assertEqual(finished['result'], {'name': finished['result']['name']})
        self.assertEqual([item[1] for item in order if item[0] == 'start'], ['a', 'b'])
        self.assertEqual(list((self.root / 'workspaces').iterdir()), [])
        self.assertEqual(self.manager.list(limit=5)[0]['state'], 'completed')

    def test_priority_orders_queued_tasks(self):
        release = threading.Event()
        order = []

        def blocker(context):
            release.wait(timeout=5)
            return {'name': 'blocker'}

        def make(name):
            def run(context):
                order.append(name)
                return {'name': name}
            return run

        first = self.manager.submit('generation', blocker)
        while self.manager.get(first['id'])['state'] != 'generating':
            time.sleep(.01)
        low = self.manager.submit('generation', make('low'), priority=-5)
        high = self.manager.submit('generation', make('high'), priority=5)
        release.set()
        self.manager.wait(first['id'], timeout=5)
        self.manager.wait(low['id'], timeout=5)
        self.manager.wait(high['id'], timeout=5)
        self.assertEqual(order, ['high', 'low'])

    def test_failure_is_persisted_with_recovery_advice_and_cleans_workspace(self):
        def blow_up(context):
            (context.workspace / 'partial.wav').write_bytes(b'x')
            raise RuntimeError('CUDA out of memory')

        job = self.manager.submit('generation', blow_up)
        finished = self.manager.wait(job['id'], timeout=5)
        self.assertEqual(finished['state'], 'failed')
        self.assertIn('内存', finished['error'])
        self.assertIn('清理', finished['error'])
        self.assertEqual(list((self.root / 'workspaces').iterdir()), [])

    def test_queued_task_can_be_cancelled_without_running(self):
        release = threading.Event()
        ran = []

        def blocker(context):
            release.wait(timeout=5)
            return {}

        def victim(context):
            ran.append(True)
            return {}

        first = self.manager.submit('generation', blocker)
        while self.manager.get(first['id'])['state'] != 'generating':
            time.sleep(.01)
        queued = self.manager.submit('generation', victim)
        self.manager.cancel(queued['id'])
        release.set()
        self.manager.wait(first['id'], timeout=5)
        finished = self.manager.wait(queued['id'], timeout=5)
        self.assertEqual(finished['state'], 'cancelled')
        self.assertEqual(ran, [])

    def test_running_task_can_be_cancelled_and_resources_are_released(self):
        started = threading.Event()

        def cooperative(context):
            started.set()
            for _ in range(500):
                if context.cancelled:
                    context.workspace.joinpath('temp.bin').write_bytes(b'x')
                    return {'partial': True}
                time.sleep(.01)
            return {'timeout': True}

        job = self.manager.submit('generation', cooperative)
        self.assertTrue(started.wait(timeout=5))
        self.manager.cancel(job['id'])
        finished = self.manager.wait(job['id'], timeout=5)
        self.assertEqual(finished['state'], 'cancelled')
        self.assertEqual(list((self.root / 'workspaces').iterdir()), [])

    def test_cleanup_callbacks_run_in_documented_order_and_failures_do_not_mask(self):
        calls = []
        register_cleanup('page-cache', 10, lambda: calls.append('page'))
        register_cleanup('history-preview', 20, lambda: calls.append('history'))

        def broken():
            raise OSError('cannot delete')
        register_cleanup('broken', 15, broken)
        released = release_low_priority_resources()
        self.assertEqual(calls, ['page', 'history'])
        self.assertIn('page-cache', released)
        self.assertIn('idle-models', released)
        # A failing cleanup is skipped instead of being reported as released.
        self.assertNotIn('broken', released)

    def test_ensure_memory_releases_then_fails_with_advice(self):
        calls = []
        register_cleanup('page-cache', 10, lambda: calls.append('page'))
        with patch('timbreforge.resource_manager.available_memory_bytes',
                   side_effect=[100, 200, 900, 900]):
            report = ensure_memory(500)
        self.assertTrue(report['ok'])
        self.assertTrue(calls)
        self.assertTrue(all(name == 'page' for name in calls))

        clear_cleanup_callbacks()
        with patch('timbreforge.resource_manager.available_memory_bytes', return_value=100):
            with patch('timbreforge.resource_manager.release_low_priority_resources',
                       return_value=[]):
                with self.assertRaises(MemoryError) as caught:
                    ensure_memory(10 ** 12, stage='生成')
        self.assertIn('内存', str(caught.exception))
        self.assertIn('重试', str(caught.exception))

    def test_available_memory_is_positive_or_unknown(self):
        value = available_memory_bytes()
        self.assertTrue(value is None or value > 0)


if __name__ == '__main__':
    unittest.main()


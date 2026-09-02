"""Smoke tests for the official local SleepFM checkpoints."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from _fixtures import make_block

from dmt_hypnodensities.sleepfm import LocalSleepFMPredictor, _torch_device


class LocalSleepFMTests(unittest.TestCase):
    def test_sleepfm_device_follows_accelerator_availability(self) -> None:
        import torch

        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(_torch_device(torch).type, expected)
        self.assertEqual(_torch_device(torch, "cpu").type, "cpu")

    def test_official_checkpoint_accepts_a_single_30_second_epoch(self) -> None:
        repository = Path(__file__).parents[1] / ".external" / "sleepfm-clinical"
        if not repository.is_dir():
            self.skipTest("The ignored local sleepfm-clinical checkout is not installed.")
        try:
            import einops  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Optional SleepFM dependencies are not installed.")

        block = make_block()
        samples = block.epochs.shape[-1]
        source_run = block.continuous_runs[0]
        one_run = replace(
            source_run,
            signal=source_run.signal[:samples],
            sample_times=source_run.sample_times[:samples],
            epochs=source_run.epochs[:1],
            end_time=float(source_run.sample_times[samples - 1]),
        )
        block = replace(
            block,
            signal=block.signal[:samples],
            sample_times=block.sample_times[:samples],
            epochs=block.epochs[:1],
            epoch_metadata=block.epoch_metadata[:1],
            continuous_runs=(one_run,),
        )
        predictor = LocalSleepFMPredictor(repository, device="cpu")
        probabilities = predictor(block, ("E6", "E8"))

        self.assertEqual(predictor._device.type, "cpu")

        self.assertEqual(probabilities.shape, (1, 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_official_checkpoint_combines_context_only_after_run_encoding(self) -> None:
        repository = Path(__file__).parents[1] / ".external" / "sleepfm-clinical"
        if not repository.is_dir():
            self.skipTest("The ignored local sleepfm-clinical checkout is not installed.")
        try:
            import einops  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Optional SleepFM dependencies are not installed.")

        block = make_block()
        samples = block.epochs.shape[-1]
        source = block.continuous_runs[0]
        first = replace(
            source,
            run_id=f"{source.run_id}_0",
            signal=source.signal[:samples],
            sample_times=source.sample_times[:samples],
            epochs=source.epochs[:1],
            end_time=float(source.sample_times[samples - 1]),
        )
        second_times = source.sample_times[samples : 2 * samples] + 5.0
        second = replace(
            source,
            run_id=f"{source.run_id}_1",
            signal=source.signal[samples : 2 * samples],
            sample_times=second_times,
            epochs=source.epochs[1:2],
            start_time=float(second_times[0]),
            end_time=float(second_times[-1]),
        )
        discontinuous = replace(
            block,
            signal=np.concatenate((first.signal, second.signal)),
            sample_times=np.concatenate((first.sample_times, second.sample_times)),
            epochs=block.epochs[:2],
            epoch_metadata=block.epoch_metadata[:2],
            continuous_runs=(first, second),
        )

        predictor = LocalSleepFMPredictor(repository, device="cpu")
        probabilities = predictor(discontinuous, ("E6",))

        self.assertEqual(probabilities.shape, (2, 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

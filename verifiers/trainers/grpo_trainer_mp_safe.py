"""grpo_trainer_mp_safe.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A thin wrapper around :class:`verifiers.trainers.grpo_trainer.GRPOTrainer` that
avoids NCCL collective time-outs in multi-GPU runs by executing
`Trainer.evaluate()` **only on the main (rank-0) process**.  All other ranks
wait at a barrier and receive the metrics via `Accelerator.broadcast_object_list`,
so logging and callbacks remain consistent.
"""

from __future__ import annotations

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from verifiers.trainers.grpo_trainer import GRPOTrainer


class GRPOMPTrainer(GRPOTrainer):
    """Multi-GPU safe GRPO trainer that gates evaluation to rank-0."""

    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        super().__init__(*args, **kwargs)
        # Use a dedicated Accelerator instance for process-group utilities.
        self._accel = Accelerator()

    # ------------------------------------------------------------------
    # Public API override
    # ------------------------------------------------------------------
    def evaluate(self, *args, **kwargs):  # noqa: D401, ANN001
        """Run `evaluate` on rank-0 only and broadcast the metrics."""

        # Rank-0 computes the metrics; others receive them.
        if self._accel.is_main_process:
            metrics = super().evaluate(*args, **kwargs)
        else:
            metrics = None  # placeholder, will be filled via broadcast

        # Broadcast the metrics dictionary so that every rank has the same data.
        # `broadcast_object_list` returns a list with the objects from rank-0.
        # Use the utility function – the Accelerator instance does not expose
        # `broadcast_object_list` as a method.
        metrics = broadcast_object_list([metrics], from_process=0)[0]
        # Ensure all processes leave this method together.
        self._accel.wait_for_everyone()
        return metrics 
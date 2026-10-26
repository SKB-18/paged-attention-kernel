"""
PagedAttention Batch Scheduler: Continuous batching with dynamic KV block budgets and preemption.
"""

from typing import List, Dict, Tuple, Optional
from collections import deque
from .sequence import Sequence, SequenceStatus
from .cache_manager import PagedKVCacheManager


class PagedScheduler:
    """
    Manages continuous batching of sequence requests against a fixed physical KV block pool.
    """

    def __init__(
        self,
        cache_manager: PagedKVCacheManager,
        max_batch_size: int = 16,
    ):
        self.cache_manager = cache_manager
        self.max_batch_size = max_batch_size

        self.waiting_queue: deque[Sequence] = deque()
        self.running_batch: List[Sequence] = []
        self.preempted_queue: deque[Sequence] = deque()

    def add_sequence(self, sequence: Sequence):
        """Submits a new sequence request to the scheduler queue."""
        sequence.status = SequenceStatus.WAITING
        self.waiting_queue.append(sequence)

    def schedule(self) -> List[Sequence]:
        """
        Schedules sequences for the current forward iteration step.

        Priority order:
        1. Keep running active sequences if blocks can be allocated.
        2. Resume preempted sequences if space permits.
        3. Admit waiting sequences from queue.
        """
        # 1. Check if all running sequences can advance
        still_running = []
        for seq in self.running_batch:
            if seq.status == SequenceStatus.FINISHED:
                self.cache_manager.free_sequence(seq.seq_id)
            else:
                still_running.append(seq)
        self.running_batch = still_running

        # 2. Try admitting waiting sequences if slots and blocks are available
        while (
            self.waiting_queue
            and len(self.running_batch) < self.max_batch_size
            and self.cache_manager.allocator.can_allocate(1)
        ):
            seq = self.waiting_queue.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running_batch.append(seq)

        return self.running_batch

    def preempt_sequence(self, sequence: Sequence):
        """Preempts a running sequence due to memory pressure."""
        self.running_batch.remove(sequence)
        sequence.status = SequenceStatus.PREEMPTED
        self.cache_manager.free_sequence(sequence.seq_id)
        self.preempted_queue.append(sequence)

    def finish_sequence(self, seq_id: int):
        """Marks a sequence as finished and releases its cache resources."""
        for seq in self.running_batch:
            if seq.seq_id == seq_id:
                seq.finish()
                self.cache_manager.free_sequence(seq_id)
                self.running_batch.remove(seq)
                break

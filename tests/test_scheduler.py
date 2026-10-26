"""
Unit tests for PagedScheduler queueing, batching, and preemption.
"""

import pytest
from serving.sequence import Sequence, SequenceStatus, SequenceGroup
from serving.cache_manager import PagedKVCacheManager
from serving.scheduler import PagedScheduler


def test_scheduler_add_and_schedule():
    """Verify scheduler admits waiting sequences up to max_batch_size."""
    manager = PagedKVCacheManager(num_blocks=32, block_size=16)
    scheduler = PagedScheduler(cache_manager=manager, max_batch_size=2)

    seq1 = Sequence(seq_id=1, prompt_token_ids=[1, 2, 3])
    seq2 = Sequence(seq_id=2, prompt_token_ids=[4, 5, 6])
    seq3 = Sequence(seq_id=3, prompt_token_ids=[7, 8, 9])

    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)
    scheduler.add_sequence(seq3)

    running = scheduler.schedule()
    assert len(running) == 2
    assert running[0].seq_id == 1
    assert running[1].seq_id == 2
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING
    assert len(scheduler.waiting_queue) == 1  # seq3 waiting


def test_scheduler_finish_and_reclamation():
    """Verify finishing a sequence frees resources and allows waiting sequences to enter."""
    manager = PagedKVCacheManager(num_blocks=32, block_size=16)
    scheduler = PagedScheduler(cache_manager=manager, max_batch_size=1)

    seq1 = Sequence(seq_id=1, prompt_token_ids=[1, 2])
    seq2 = Sequence(seq_id=2, prompt_token_ids=[3, 4])

    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)

    running = scheduler.schedule()
    assert len(running) == 1
    assert running[0].seq_id == 1

    # Finish seq1
    scheduler.finish_sequence(seq_id=1)
    assert seq1.status == SequenceStatus.FINISHED

    # Next schedule step admits seq2
    running = scheduler.schedule()
    assert len(running) == 1
    assert running[0].seq_id == 2


def test_sequence_group_branching():
    """Verify SequenceGroup managing multiple branches from a single prompt."""
    group = SequenceGroup(group_id=1, prompt_token_ids=[10, 20, 30])
    assert len(group) == 1
    assert not group.is_finished

    # Add 2 parallel sampling branches
    child1 = group.add_branch(child_seq_id=2)
    child2 = group.add_branch(child_seq_id=3)

    assert len(group) == 3
    assert child1.prompt_token_ids == [10, 20, 30]
    assert child2.prompt_token_ids == [10, 20, 30]

    # Complete all branches
    for s in group.sequences:
        s.finish()
    assert group.is_finished


def test_scheduler_preempt_sequence():
    """Verify preemption changes sequence status and moves it to preempted queue."""
    manager = PagedKVCacheManager(num_blocks=10, block_size=16)
    scheduler = PagedScheduler(cache_manager=manager, max_batch_size=2)

    seq1 = Sequence(seq_id=1, prompt_token_ids=[1, 2])
    scheduler.add_sequence(seq1)
    scheduler.schedule()
    assert seq1 in scheduler.running_batch

    scheduler.preempt_sequence(seq1)
    assert seq1 not in scheduler.running_batch
    assert seq1.status == SequenceStatus.PREEMPTED
    assert seq1 in scheduler.preempted_queue

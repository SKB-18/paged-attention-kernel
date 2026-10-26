"""
Sequence and SequenceGroup abstractions for LLM request lifecycle tracking.
"""

from enum import Enum
from typing import List, Optional


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    FINISHED = "FINISHED"


class Sequence:
    """
    Represents an individual generation stream with prompt and generated tokens.
    """

    def __init__(self, seq_id: int, prompt_token_ids: List[int]):
        self.seq_id = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.status = SequenceStatus.WAITING

    @property
    def total_len(self) -> int:
        """Total sequence length (prompt + generated tokens)."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def append_token(self, token_id: int):
        """Appends a newly decoded token to this sequence."""
        self.output_token_ids.append(token_id)

    def finish(self):
        """Marks sequence as complete."""
        self.status = SequenceStatus.FINISHED

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.value}, "
            f"prompt_len={self.prompt_len}, gen_len={self.output_len}, total={self.total_len})"
        )


class SequenceGroup:
    """
    Represents a prompt and all its associated parallel sampling or beam search branches.
    """

    def __init__(self, group_id: int, prompt_token_ids: List[int]):
        self.group_id = group_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.sequences: List[Sequence] = [Sequence(seq_id=group_id, prompt_token_ids=prompt_token_ids)]

    @property
    def is_finished(self) -> bool:
        """True if all child sequences in this group have completed."""
        return all(seq.status == SequenceStatus.FINISHED for seq in self.sequences)

    def add_branch(self, child_seq_id: int) -> Sequence:
        """Creates a new sequence branch inheriting parent prompt and current tokens."""
        parent = self.sequences[0]
        child = Sequence(seq_id=child_seq_id, prompt_token_ids=parent.prompt_token_ids)
        child.output_token_ids = list(parent.output_token_ids)
        child.status = parent.status
        self.sequences.append(child)
        return child

    def __len__(self) -> int:
        return len(self.sequences)

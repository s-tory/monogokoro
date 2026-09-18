# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-episode salience tags, written beside the dataset as the episodes are recorded.

Whether an episode went well, was a near miss, or was ordinary can only be recorded
live. The operator attaches it while their hands are still on the leader arm, and no
amount of later review reconstructs it: what is being recorded is their judgement at
that moment, not a property of the frames.

Every saved episode gets exactly one tag and the tags are written in episode order, so
the sidecar carries no indices and no timestamps -- line N is episode N. That only holds
because *ordinary* has a key of its own. With "unmarked means ordinary", "there was
nothing to flag" and "nobody reached the keyboard in time" are the same absence in the
file and cannot be told apart afterwards.

Two further tags cover the cases where no judgement was given. They are kept distinct
from ``ORDINARY`` for exactly that reason -- folding them into it would be inventing a
judgement nobody made:

* ``UNLABELLED`` -- the episode ended on the clock rather than on a key, or it was
  recorded before this file existed.
* ``QUIT`` -- the operator stopped the session during the episode. The recording loop
  still saves that episode, so it needs a tag.

The tag is deliberately not a dataset column. Float columns are bundled into
``observation.state``, so a per-frame column would reach the policy's input; the
salience is a label *about* the episode, for weighting it during training, and it
belongs outside the frames.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

SALIENCE_FILENAME = "salience.txt"

GOOD = "g"  # it worked
NEAR_MISS = "b"  # that was close
ORDINARY = "→"  # nothing to say about it (right arrow, the "just go on" key)
QUIT = "q"  # the session was stopped during this episode
UNLABELLED = "?"  # ended on the clock, or recorded before tags existed

#: Tags a key press can produce. ``UNLABELLED`` is not here: it is never chosen, only
#: written when nothing was chosen.
KEYED_TAGS = (GOOD, NEAR_MISS, ORDINARY, QUIT)


def salience_path(root: str | Path) -> Path:
    """Return the sidecar path for a dataset rooted at ``root``."""
    return Path(root) / SALIENCE_FILENAME


def read_salience(root: str | Path) -> list[str]:
    """Return the tags recorded so far, in episode order (empty when none are)."""
    path = salience_path(root)
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").split()


def append_salience(root: str | Path, tag: str) -> None:
    """Append one episode's tag, flushed to disk before returning.

    The flush is not caution about crashes in general: this tag is the one thing in the
    recording that cannot be recovered by re-reading the episode, so losing it costs the
    episode's label permanently. Episodes are seconds apart, so the fsync costs nothing
    that matters.
    """
    path = salience_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{tag}\n")
        f.flush()
        os.fsync(f.fileno())


def align_salience(root: str | Path, num_episodes: int) -> None:
    """Pad the sidecar with ``UNLABELLED`` so line N is episode N before appending.

    Called once when recording starts on a dataset that already holds episodes (a resume,
    or a dataset recorded before this sidecar existed). Padding is the only way to keep
    the positional contract without inventing labels -- ``UNLABELLED`` states that nobody
    judged those episodes, which is true.

    A sidecar *longer* than the dataset is not repaired here: it means the file and the
    dataset disagree about what was recorded, and guessing which is right would destroy
    whichever one is.
    """
    existing = read_salience(root)
    missing = num_episodes - len(existing)
    if missing > 0:
        for _ in range(missing):
            append_salience(root, UNLABELLED)
        logger.warning(
            "%s had %d tags for %d episodes; padded %d with %r (nobody judged those).",
            salience_path(root),
            len(existing),
            num_episodes,
            missing,
            UNLABELLED,
        )
    elif missing < 0:
        logger.error(
            "%s holds %d tags but the dataset has %d episodes. Line N no longer means "
            "episode N. Not repairing it -- check the file by hand.",
            salience_path(root),
            len(existing),
            num_episodes,
        )

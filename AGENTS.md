This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Principles

This fork exists to build a reflex layer, and it is being built on four observations that
predate the field by 2500 years. They are not decoration. Each one is a rule about what to
write and — more often — what to refuse to write.

- **諸行無常 / anicca — nothing holds still.** Every measured constant decays. Gains, offsets,
  calibration, latency, the droop of a servo under load: each is a snapshot of one machine on
  one day. Date every number you record and name the rig it came from. If a constant's
  provenance is unverifiable, retake it — never inherit it.
- **一切皆苦 / dukkha — the error never reaches zero.** A controller with no standing error is
  a controller that has stopped. Do not design for the fixed point; design for the size and
  shape of the residual. Report the error that remains, always, next to the improvement.
- **縁起 / pratītyasamutpāda — nothing means anything alone.** A policy's output is not a
  property of the policy. It is a property of policy × body × load × context. A number measured
  without its conditions is not a weak measurement, it is not a measurement. Publish the
  conditions alongside it, or publish nothing.
- **無記 / avyākata — refuse the question that changes nothing.** The hardest discipline here.
  When something has not been measured, the correct output is silence, not a plausible number.
  When a question cannot alter what we build next, decline it and say why. Withdrawing a claim
  that turned out to be unmeasured is normal maintenance, not failure — do it without being
  asked.

Practical consequence, in one line: **measure first, and say only what the measurement says.**

## Method

The four above say what not to write. These say where to look first. They came out of this
project's own mistakes, but none of them are specific to it.

- **Ask biology before comparing engineering options.** When an architecture has more than one
  defensible shape, look at how the animal solves it before weighing the shapes against each
  other. The load-bearing half is what the biological structure does _not_ do — finding that the
  pontine nuclei compute nothing killed a whole branch of the design in one step. And when the
  animal has already answered, say so and move on; do not hand the user back a two-option
  question you already know the answer to.
- **Convergence is evidence; divergence is a clue.** Fields that never cite each other land on
  the same mechanism under different names — the cerebellum's silent-climbing-fibre gate is
  control theory's conditional integration / anti-windup. If you are stuck, search under the
  other field's name; the prior art is usually there. When two fields _disagree_, the useful
  question is not which is right but which constraint differs. (Joints are lubricated, so
  biology has almost no static friction and cannot be copied on friction — that is where a
  measured deadband has to come from instead.)
- **The measurement is never wrong; the question was.** A run that returns a surprising number
  returned a correct answer to whatever you actually asked. Before doubting the instrument,
  check what you asked. Corollary on who judges: the payment for being right is prediction, and
  the judge is nature, not the audience. Rejection carries no information in either direction —
  Semmelweis had the measurement, published it, and was not believed.
- **Write the test that can kill the hypothesis before writing the implementation.** The cost of
  a wrong hypothesis is not the wrongness, it is the code built before it was checked. Four
  plausible hypotheses in a row were wrong here and cost nothing; one of them cost a shader, a
  GPU buffer, a CPU reference path and a test suite, all deleted by a three-line check that
  could have run first. Read-only measurement needs no permission and no plan — run it the
  moment you think of it. Anything that moves hardware does need one.
- **Do not replace a proven tool with an unverified script.** Before proposing the replacement,
  check whether the real reason is that the existing tool is interactive and you cannot drive
  it yourself. That is your convenience, not the user's safety, and the two must not be mixed.
  Weigh it against how often the task runs — a once-ever setup step does not earn new code.
  A safety check that has never been exercised is not a safety check.
- **A metric is not trusted until it tracks a human's blind judgment.** The minimum-jerk model
  is not why human reaching looks smooth; it is what was found by looking for a quantity that
  matched what people already saw. The eye comes first and validates the number, not the other
  way round — so show the comparison blind, then keep only the metrics that followed the choice.
  And one number cannot answer two questions whose window lengths disagree; when that happens,
  report two.
- **Documentation is a correctness gate, not an append gate.** Before pushing, grep the docs for
  the claims your change just touched and ask whether any of them became false. Adding a section
  for every change is how docs stop being readable; deleting and correcting is the normal case.
  When you withdraw a number, leave what it used to say and why it was wrong — readers remember
  the old version.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).

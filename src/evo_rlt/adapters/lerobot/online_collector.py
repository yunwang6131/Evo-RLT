from __future__ import annotations

import torch

from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.adapters.lerobot.record.annotations import SOURCE_HUMAN


class RLTOnlineCollector:
    """Accumulates per-frame robot data into ChunkTransitions every C frames.

    Transitions stage locally per-episode and only commit to the global
    replay buffer on flush_episode() -- see _episode_staging.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        chunk_length: int,
        action_dim: int,
    ):
        self._buffer = replay_buffer
        self._C = chunk_length
        self._action_dim = action_dim
        self._frame_actions: list[torch.Tensor] = []
        self._frame_sources: list[float] = []
        self._chunk_state: torch.Tensor | None = None
        self._chunk_ref: torch.Tensor | None = None
        self._chunk_is_critical: float = 0.0
        self._episode_id: int = -1
        self._prev_transition: ChunkTransition | None = None
        # Transitions accumulate here, NOT in the global replay buffer,
        # until flush_episode() commits them. A rerecorded/discarded/never-
        # labeled episode's staged transitions are simply dropped by the next
        # start_episode() call instead of permanently polluting the global
        # buffer with dangling non-terminal transitions (no valid next_state,
        # no outcome label, possibly built from footage the user rejected).
        self._episode_staging: list[ChunkTransition] = []
        # True once flush_episode() has fired for this "sub-episode" (one
        # critical-phase attempt). The recorded dataset episode may continue
        # past that point (e.g. VLA autonomously finishing a subsequent
        # placement step) -- on_frame() ignores those later frames entirely
        # so the RL reward reflects only what the actor actually controlled,
        # not whatever happens afterward. Reset by the next start_episode().
        self._flushed: bool = False

    def start_episode(self, episode_id: int) -> None:
        self._episode_id = episode_id
        self._frame_actions.clear()
        self._frame_sources.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        self._prev_transition = None
        self._episode_staging = []
        self._flushed = False

    def on_frame(
        self,
        action: torch.Tensor,
        state_vec: torch.Tensor | None,
        ref_chunk: torch.Tensor | None,
        source_type: float,
        is_critical: float,
    ) -> ChunkTransition | None:
        if self._flushed:
            return None
        if len(self._frame_actions) == 0:
            # Capture state at chunk start. During intervention state_vec may be None;
            # fall back to the previous transition's state so human chunks are not dropped.
            if state_vec is not None:
                self._chunk_state = state_vec
                self._chunk_ref = ref_chunk
            elif self._prev_transition is not None:
                self._chunk_state = self._prev_transition.next_state_vec
                self._chunk_ref = self._prev_transition.next_ref_chunk
            self._chunk_is_critical = is_critical

        self._frame_actions.append(action.detach().cpu())
        self._frame_sources.append(source_type)

        if len(self._frame_actions) >= self._C:
            return self._emit_transition(done=False)
        return None

    def flush_episode(self, episode_success: bool) -> ChunkTransition | None:
        """Finalize the critical-phase attempt, marking the terminal
        transition done=1 and writing the sparse binary reward (r=1 iff
        success, else 0) onto its last valid timestep, matching the paper's
        terminal-reward-only setup. Commits every transition staged so far to
        the global replay buffer, then ignores any further on_frame() calls
        until the next start_episode() -- call this at the moment the
        critical phase itself resolves (success/failure), not necessarily at
        whole-episode end: the recorded episode may continue afterward (e.g.
        VLA autonomously finishing a subsequent step), but that tail is
        dataset-only and must not leak into the RL reward. For a
        rerecorded/discarded/never-labeled attempt, just don't call this and
        let the next start_episode() drop the staged data instead.
        """
        terminal_reward = 1.0 if episode_success else 0.0
        result: ChunkTransition | None = None
        if self._frame_actions:
            result = self._emit_transition(done=True, terminal_reward=terminal_reward)
        elif self._prev_transition is not None:
            # Episode length was exact multiple of C — mark last staged chunk as terminal.
            self._prev_transition.done = torch.tensor(1.0)
            actual = int(self._prev_transition.actual_steps.item())
            self._prev_transition.reward_seq[actual - 1] = terminal_reward
        for transition in self._episode_staging:
            self._buffer.add(transition)
        self._episode_staging = []
        self._flushed = True
        return result

    def _emit_transition(self, done: bool, terminal_reward: float = 0.0) -> ChunkTransition | None:
        if self._chunk_state is None or self._chunk_ref is None:
            self._frame_actions.clear()
            self._frame_sources.clear()
            self._chunk_is_critical = 0.0
            return None

        actual = len(self._frame_actions)
        exec_list = self._frame_actions[: self._C]
        exec_chunk = torch.stack(exec_list)
        if actual < self._C:
            pad = torch.zeros(self._C - actual, self._action_dim, dtype=exec_chunk.dtype)
            exec_chunk = torch.cat([exec_chunk, pad])

        # Deterministic tie-break: human wins ties (highest priority)
        dominant_source = max(set(self._frame_sources), key=lambda s: (self._frame_sources.count(s), s))
        ref = self._chunk_ref.cpu()
        if dominant_source == SOURCE_HUMAN:
            ref = exec_chunk.clone()

        state = self._chunk_state.cpu()
        reward_seq = torch.zeros(self._C)
        if done and actual > 0:
            reward_seq[actual - 1] = terminal_reward
        transition = ChunkTransition(
            state_vec=state,
            exec_chunk=exec_chunk,
            ref_chunk=ref,
            reward_seq=reward_seq,
            next_state_vec=state,
            next_ref_chunk=ref,
            done=torch.tensor(float(done)),
            intervention=torch.tensor(float(dominant_source == SOURCE_HUMAN)),
            actual_steps=torch.tensor(actual),
            source=torch.tensor(int(dominant_source)),
            episode_id=torch.tensor(self._episode_id),
            is_critical=torch.tensor(self._chunk_is_critical),
        )

        if self._prev_transition is not None:
            self._prev_transition.next_state_vec = state.clone()
            self._prev_transition.next_ref_chunk = ref.clone()

        self._episode_staging.append(transition)
        self._prev_transition = transition

        self._frame_actions.clear()
        self._frame_sources.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        return transition

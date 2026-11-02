"""DQN on candidate-link states with epsilon-greedy binary actions.

Important: Candidate classification supplies no action-controlled future transition.
The defensible default gamma=0 is a one-step contextual bandit, a special case
of Q-learning. A nonzero gamma is an observational chain, NOT proof of
long-horizon optimal link-recommendation policy.
"""
from __future__ import annotations
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, states):
        return self.net(states)


class TransitionMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def add_many(self, items):
        self.memory.extend(items)

    def sample(self, rng, size):
        if not self.memory:
            return []
        ids = rng.choice(len(self.memory), size=min(size, len(self.memory)), replace=False)
        return [self.memory[int(i)] for i in np.atleast_1d(ids)]

    def __len__(self):
        return len(self.memory)


def build_transitions(q, states: torch.Tensor, labels: torch.Tensor,
                      epsilon: float, rng):
    """Feedback from LABELS REVEALED after a task, never during prequential eval."""
    if not len(states):
        return []
    with torch.no_grad():
        qs = q(states).cpu().numpy()
    actions = qs.argmax(1)
    explore = rng.random(len(actions)) < epsilon
    actions[explore] = rng.integers(0, 2, size=int(explore.sum()))
    yy = labels.long().cpu().numpy()
    reward = np.where(actions == yy, 1.0, -1.0)
    states_cpu = states.detach().cpu()
    out = []
    for i in range(len(states)):
        next_state = states_cpu[i + 1] if i + 1 < len(states) else torch.zeros_like(states_cpu[i])
        out.append((states_cpu[i], int(actions[i]), float(reward[i]), next_state,
                    float(i == len(states) - 1)))
    return out


def _batch(transitions, device):
    return (torch.stack([x[0] for x in transitions]).to(device),
            torch.tensor([x[1] for x in transitions], dtype=torch.long, device=device),
            torch.tensor([x[2] for x in transitions], dtype=torch.float32, device=device),
            torch.stack([x[3] for x in transitions]).to(device),
            torch.tensor([x[4] for x in transitions], dtype=torch.float32, device=device))


def td_loss(q, target, transitions, gamma, device):
    states, actions, rewards, next_states, terminal = _batch(transitions, device)
    chosen = q(states).gather(1, actions[:, None]).flatten()
    with torch.no_grad():
        boot = target(next_states).max(1).values
        td_target = rewards + gamma * (1 - terminal) * boot
    return F.smooth_l1_loss(chosen, td_target)


def train_dqn(q, optimizer, states, labels, memory, rng, ewc=None,
              ewc_lambda=1.0, epochs=12, batch_size=64, epsilon=0.25,
              gamma=0.0, replay=True):
    from copy import deepcopy
    device = next(q.parameters()).device
    target = deepcopy(q).eval()
    transitions = build_transitions(q, states, labels, epsilon, rng)
    if not transitions:
        return float("nan"), []
    last = float("nan")
    for e in range(epochs):
        rng.shuffle(transitions)
        for begin in range(0, len(transitions), batch_size):
            batch = transitions[begin:begin + batch_size]
            if replay:
                batch = batch + memory.sample(rng, batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss = td_loss(q, target, batch, gamma, device)
            if ewc is not None:
                loss = loss + ewc_lambda * ewc.penalty(q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 5)
            optimizer.step()
            last = float(loss.detach())
        if (e + 1) % 3 == 0:
            target.load_state_dict(q.state_dict())
    # Past transitions are retained only after the current task was revealed.
    memory.add_many(transitions)
    return last, transitions


def consolidate_dqn(q, transitions, ewc, gamma=0.0, max_examples=128, batch_size=16):
    if not transitions:
        return
    # Target frozen to current network when measuring observed TD gradients.
    from copy import deepcopy
    target = deepcopy(q).eval()
    selection = transitions[:max_examples]
    device = next(q.parameters()).device
    functions = []
    for begin in range(0, len(selection), batch_size):
        batch = selection[begin:begin+batch_size]
        functions.append(lambda batch=batch: td_loss(q, target, batch, gamma, device))
    ewc.consolidate(q, functions)


def q_scores(q, states):
    q.eval()
    with torch.no_grad():
        prediction = q(states)
        return (prediction[:, 1] - prediction[:, 0]).cpu().numpy()

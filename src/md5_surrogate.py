"""Differentiable MD5 surrogate — learns the forward pass round by round.

Instead of learning the end-to-end Jacobian (which requires a shallow network
to shortcut a deep computation), learn MD5 itself: one neural network layer
per MD5 round, each learning a simple local transform. The Jacobian falls
out of autograd for free.

Each MD5 round is:
    f = F(b,c,d)                    # simple nonlinear function
    temp = a + f + K[i] + M[g(i)]   # additions
    (a,b,c,d) = (d, b+rot(temp,s), b, c)

One MLP can learn this. Stack 64 = learn MD5. Train with teacher forcing:
each round layer gets the *real* intermediate state as input (from integer
MD5) and learns to predict the next state. At inference, rounds chain
together and the autograd Jacobian is exact through the learned forward pass.
"""

import torch
import torch.nn as nn

from .md5_gpu import _SHIFT


class RoundLayer(nn.Module):
    """Learns one MD5 round: (state, message_word, round_info) -> new_state.

    State is 16 bytes (4 words × 4 bytes), message word is 4 bytes.
    """

    def __init__(self, d_hidden: int = 256):
        super().__init__()
        # Input: 16 (state bytes) + 4 (message word bytes) + 2 (round index, shift amount)
        # Output: 16 (new state bytes)
        self.net = nn.Sequential(
            nn.Linear(22, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 16),
        )

    def forward(self, state: torch.Tensor, msg_word: torch.Tensor,
                round_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, 16) float — 4 words as 16 little-endian bytes, normalized to [0,1]
            msg_word: (B, 4) float — 1 word as 4 bytes, normalized to [0,1]
            round_info: (B, 2) float — [round_index/64, shift_amount/25]
        Returns:
            (B, 16) float — predicted next state bytes, normalized
        """
        x = torch.cat([state, msg_word, round_info], dim=-1)
        return self.net(x)


class MD5Surrogate(nn.Module):
    """Differentiable MD5 surrogate with per-round learned layers.

    Args:
        num_rounds: MD5 rounds to model (1-64)
        d_hidden: hidden dim for each round's MLP
        shared_weights: if True, all rounds share the same MLP
            (smaller model, but each round has unique constants so
            the round_info input disambiguates)
    """

    def __init__(self, num_rounds: int = 64, d_hidden: int = 256,
                 shared_weights: bool = False):
        super().__init__()
        self.num_rounds = num_rounds

        if shared_weights:
            layer = RoundLayer(d_hidden)
            self.rounds = nn.ModuleList([layer] * num_rounds)
        else:
            self.rounds = nn.ModuleList([
                RoundLayer(d_hidden) for _ in range(num_rounds)
            ])

        # MD5 message schedule: which word index is used at each round
        self._schedule = []
        for i in range(64):
            if i < 16:
                self._schedule.append(i)
            elif i < 32:
                self._schedule.append((5 * i + 1) % 16)
            elif i < 48:
                self._schedule.append((3 * i + 5) % 16)
            else:
                self._schedule.append((7 * i) % 16)

    def forward(self, message_bytes: torch.Tensor,
                initial_state: torch.Tensor | None = None) -> torch.Tensor:
        """Run the surrogate forward pass.

        Args:
            message_bytes: (B, 64) float tensor — message bytes normalized to [0,1]
            initial_state: (B, 16) float tensor — initial state bytes normalized to [0,1]
                If None, uses MD5's initial state.
        Returns:
            (B, 16) float — predicted final hash state bytes, normalized to [0,1]
        """
        B = message_bytes.shape[0]
        device = message_bytes.device

        if initial_state is None:
            # MD5 initial state as bytes: 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476
            init_bytes = [
                0x01, 0x23, 0x45, 0x67,  # a0 little-endian
                0x89, 0xAB, 0xCD, 0xEF,  # b0
                0xFE, 0xDC, 0xBA, 0x98,  # c0
                0x76, 0x54, 0x32, 0x10,  # d0
            ]
            initial_state = torch.tensor(
                init_bytes, dtype=torch.float32, device=device
            ).unsqueeze(0).expand(B, -1) / 255.0

        # Reshape message to 16 words of 4 bytes each
        msg_words = message_bytes.reshape(B, 16, 4)  # (B, 16, 4)

        state = initial_state
        for i in range(self.num_rounds):
            g = self._schedule[i]
            word = msg_words[:, g, :]  # (B, 4)
            round_info = torch.tensor(
                [i / 64.0, _SHIFT[i] / 25.0],
                device=device, dtype=torch.float32
            ).unsqueeze(0).expand(B, -1)
            state = self.rounds[i](state, word, round_info)

        return state

    def forward_with_intermediates(self, message_bytes: torch.Tensor,
                                   initial_state: torch.Tensor | None = None,
                                   ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning intermediate states at each round.

        Returns:
            final_state: (B, 16) float
            intermediates: list of (B, 16) float, length = num_rounds + 1
        """
        B = message_bytes.shape[0]
        device = message_bytes.device

        if initial_state is None:
            init_bytes = [
                0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
            ]
            initial_state = torch.tensor(
                init_bytes, dtype=torch.float32, device=device
            ).unsqueeze(0).expand(B, -1) / 255.0

        msg_words = message_bytes.reshape(B, 16, 4)
        state = initial_state
        intermediates = [state]

        for i in range(self.num_rounds):
            g = self._schedule[i]
            word = msg_words[:, g, :]
            round_info = torch.tensor(
                [i / 64.0, _SHIFT[i] / 25.0],
                device=device, dtype=torch.float32
            ).unsqueeze(0).expand(B, -1)
            state = self.rounds[i](state, word, round_info)
            intermediates.append(state)

        return state, intermediates


def words_to_state_bytes(states: torch.Tensor) -> torch.Tensor:
    """Convert (B, 4) int64 word states to (B, 16) float byte states normalized to [0,1].

    For converting md5_gpu intermediate states to surrogate training targets.
    """
    result = []
    for i in range(4):
        w = states[:, i]
        result.append((w & 0xFF).float())
        result.append(((w >> 8) & 0xFF).float())
        result.append(((w >> 16) & 0xFF).float())
        result.append(((w >> 24) & 0xFF).float())
    return torch.stack(result, dim=1) / 255.0

import torch

from mtp.models.loss import IGNORE_TOKEN_ID
from mtp.utils.packing import packed_targets_to_target_windows


def test_expand_target_windows_of_packed_targets():
    EOS = -1
    yy = torch.tensor([[1, 2, 3, EOS, 1, 2, EOS],
                       [1, 2, 3, 4, 5, 6, EOS]],
                       dtype=torch.int)

    outs = packed_targets_to_target_windows(yy, 3, EOS, IGNORE_TOKEN_ID)

    expected_output = torch.tensor([
        [[1, 2, 3],
         [2, 3, EOS],
         [3, EOS, IGNORE_TOKEN_ID],
         [EOS, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID],
         [1, 2, EOS],
         [2, EOS, IGNORE_TOKEN_ID],
         [EOS, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID]],
        [[1, 2, 3],
         [2, 3, 4],
         [3, 4, 5],
         [4, 5, 6],
         [5, 6, EOS],
         [6, EOS, IGNORE_TOKEN_ID],
         [EOS, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID]]
        ], dtype=torch.int)

    assert torch.allclose(outs, expected_output)

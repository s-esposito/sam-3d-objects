import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from sea_raft_core.raft import RAFT
from sea_raft_core.utils.utils import json_to_args


@pytest.fixture(scope="module")
def flow_model():
    ckpt_path = "sea_raft_core/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth"
    json_path = "sea_raft_core/configs/spring-L.json"
    args = json_to_args(json_path)

    model = RAFT(args)
    model.load_ckpt(ckpt_path)
    model.cuda()
    model.eval()
    return model


def test_calc_forward_and_backward_flow(flow_model):
    cur_rgb = torch.rand([2, 3, 1080, 1920], dtype=torch.float32).cuda()
    prev_rgb = torch.rand([2, 3, 1080, 1920], dtype=torch.float32).cuda()

    with torch.no_grad():
        cur_to_prev_flow, _ = flow_model.calc_flow(cur_rgb, prev_rgb)
        prev_to_cur_flow, _ = flow_model.calc_flow(prev_rgb, cur_rgb)

    assert cur_to_prev_flow.shape == torch.Size([2, 2, 1080, 1920])
    assert prev_to_cur_flow.shape == torch.Size([2, 2, 1080, 1920])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

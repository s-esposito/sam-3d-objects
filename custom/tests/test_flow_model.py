import sys
from pathlib import Path

# Add paths for imports
custom_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(custom_dir))
sys.path.insert(0, str(custom_dir / "submodules" / "sea_raft_core"))

import pytest
import torch
import numpy as np
from PIL import Image
from raft import RAFT
from utils.utils import json_to_args
from utils.flow_viz import flow_to_image


@pytest.fixture(scope="module")
def flow_model():
    ckpt_path = "submodules/sea_raft_core/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth"
    json_path = "submodules/sea_raft_core/configs/spring-L.json"
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


def test_flow_with_real_images(flow_model):
    """Test flow computation with real images and save visualizations."""
    # Load test images
    assets_dir = Path(__file__).parent / "assets"
    output_dir = assets_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    img0_path = assets_dir / "img_flow_0.png"
    img1_path = assets_dir / "img_flow_1.png"
    
    # Load and preprocess images
    img0 = np.array(Image.open(img0_path).convert("RGB"))
    img1 = np.array(Image.open(img1_path).convert("RGB"))
    
    print(f"Image 0 shape: {img0.shape}")
    print(f"Image 1 shape: {img1.shape}")
    
    # Convert to torch tensors [B, C, H, W] and normalize to [0, 255] range (RAFT expects this)
    img0_t = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float().cuda()
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().cuda()
    
    print(f"Tensor 0 shape: {img0_t.shape}")
    print(f"Tensor 1 shape: {img1_t.shape}")
    
    # Compute forward and backward flow
    with torch.no_grad():
        flow_0_to_1, _ = flow_model.calc_flow(img0_t, img1_t)  # flow from img0 to img1
        flow_1_to_0, _ = flow_model.calc_flow(img1_t, img0_t)  # flow from img1 to img0
    
    print(f"Flow 0->1 shape: {flow_0_to_1.shape}")
    print(f"Flow 1->0 shape: {flow_1_to_0.shape}")
    
    # Convert flows to numpy [H, W, 2] for visualization
    flow_0_to_1_np = flow_0_to_1[0].permute(1, 2, 0).cpu().numpy()
    flow_1_to_0_np = flow_1_to_0[0].permute(1, 2, 0).cpu().numpy()
    
    # Visualize flows
    flow_0_to_1_vis = flow_to_image(flow_0_to_1_np)
    flow_1_to_0_vis = flow_to_image(flow_1_to_0_np)
    
    # Save visualizations
    Image.fromarray(flow_0_to_1_vis.astype(np.uint8)).save(output_dir / "flow_0_to_1.png")
    Image.fromarray(flow_1_to_0_vis.astype(np.uint8)).save(output_dir / "flow_1_to_0.png")
    
    print(f"Saved flow visualizations to {output_dir}")
    
    # Basic assertions
    assert flow_0_to_1.shape[0] == 1  # batch size
    assert flow_0_to_1.shape[1] == 2  # u, v channels
    assert flow_0_to_1.shape[2] == img0.shape[0]  # height
    assert flow_0_to_1.shape[3] == img0.shape[1]  # width


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_flow_with_real_images"])

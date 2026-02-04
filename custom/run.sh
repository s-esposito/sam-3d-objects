# DAVIS scripts
# python custom/demo.py --dataset davis --scene-name car-turn
# python custom/redecode.py --dataset davis --scene-name car-turn --average-frames --weighted-average
# python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 --canonicalization average --weighting-type mask-error --refine-poses --save-renders --save-metrics

# per-frame, no refining
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics

# per-frame, pose refining (per-frame scale)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --refine-poses --refine-scale perframe

# canonical (averaged), no refining, with mask-error weighting
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --canonicalization average --weighting-type mask-error

# canonical (averaged), pose refining, with mask-error weighting (per-frame scale)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --refine-poses --refine-scale perframe \
    --canonicalization average --weighting-type mask-error

# canonical (pickone from frame 0), no refining
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --canonicalization pickone --canon-frame 0

# canonical (pickone auto - best coverage per object), no refining
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --canonicalization pickone

# canonical (pickone from frame 0), pose refining
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --refine-poses --refine-scale perframe \
    --canonicalization pickone --canon-frame 0

# per-frame, pose refining (median scale as init, not refined)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --median-scale --refine-scale none --refine-poses

# per-frame, pose refining (median scale as init, perframe refined)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --median-scale --refine-scale perframe --refine-poses

# per-frame, pose refining (median scale as init, global refined)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --median-scale --refine-poses --refine-scale global --refine-batch-size 0 \
    --canonicalization pickone

# canonical (averaged), pose refining, with mask-error weighting (global scale optimization)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --refine-poses --refine-scale global --refine-batch-size 4 \
    --canonicalization average --weighting-type mask-error

# per-frame, pose refining with optical flow correspondence loss
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --refine-poses --refine-scale perframe \
    --use-flow --flow-weight 0.1

# canonical (averaged), pose refining with flow loss and global scale
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --refine-poses --refine-scale global \
    --canonicalization average --weighting-type mask-error \
    --use-flow --flow-weight 0.1

# CURRENT TEST SCRIPT
python custom/evaluate_sequence.py \
    --dataset davis \
    --scene-name car-turn \
    --frame-stride 10 \
    --save-renders \
    --save-metrics \
    --refine-poses \
    --canonicalization none \
    --refine-config custom/configs/refinement.yaml

# Example: disable background gaussians
# python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --no-background

# Example: force full inference (ignore cache)
# python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --no-use-cache

# Visualize the temporal point cloud
python custom/visualize_temporal_pointcloud.py \
    custom/results/davis/eval/point_clouds/car-turn_temporal.npz

# Show all frames at once with time coloring
python custom/visualize_temporal_pointcloud.py \
    custom/results/davis/eval/point_clouds/car-turn_temporal.npz --show-all

# Filter by opacity
python custom/visualize_temporal_pointcloud.py \
    custom/results/davis/eval/point_clouds/car-turn_temporal.npz --opacity-threshold 0.5

# # Kubric scripts
# python custom/demo.py --dataset kubric4d --scene-name scn02719 --object-index 15 --frame-index 0
# # python custom/redecode.py --dataset kubric4d --scene-name scn02719 --object-index 15 --frame-index 0 
# python custom/render_scene.py --dataset kubric4d --scene-name scn02719 --image-size 1024 --distance 2 --fov 40 --object-index 15 --frame-index 0 
# # python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --frame-stride 10 --canonicalization average --weighting-type mask-error --refine-poses --save-renders --save-metrics

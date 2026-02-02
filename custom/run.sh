# DAVIS scripts
# python custom/demo.py --dataset davis --scene-name car-turn
# python custom/redecode.py --dataset davis --scene-name car-turn --average-frames --weighted-average
# python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 --average-tokens --weighting-type mask-error --refine-poses --save-renders --save-metrics

# per-frame, no refining
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics

# per-frame, pose refining (per-frame scale)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --refine-poses --refine-scale perframe

# per-frame, no refining, with mask-error weighting averaging
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --average-tokens --weighting-type mask-error

# per-frame, pose refining, with mask-error weighting averaging (per-frame scale)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --refine-poses --refine-scale perframe \
    --average-tokens --weighting-type mask-error


# per-frame, pose refining (global scale, not refined)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --median-scale --refine-scale none --refine-poses 

# per-frame, pose refining (global scale, perframe refined)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --median-scale --refine-scale perframe --refine-poses 

# per-frame, pose refining, with mask-error weighting averaging (global scale, perframe refined)
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 \
    --save-renders --save-metrics \
    --median-scale --refine-scale perframe --refine-poses \
    --average-tokens --weighting-type mask-error

# # Kubric scripts
# python custom/demo.py --dataset kubric4d --scene-name scn02719 --object-index 15 --frame-index 0
# # python custom/redecode.py --dataset kubric4d --scene-name scn02719 --object-index 15 --frame-index 0 
# python custom/render_scene.py --dataset kubric4d --scene-name scn02719 --image-size 1024 --distance 2 --fov 40 --object-index 15 --frame-index 0 
# # python custom/evaluate_sequence.py --dataset kubric4d --scene-name scn02719 --frame-stride 10 --average-tokens --weighting-type mask-error --refine-poses --save-renders --save-metrics

# python custom/demo.py --dataset davis --scene-name car-turn
# python custom/redecode.py --dataset davis --scene-name car-turn --average-frames --weighted-average
# python custom/render_scene.py --dataset davis --scene-name car-turn --mode averaged --image-size 1024 --distance 2 --fov 40
python custom/evaluate_sequence.py --dataset davis --scene-name car-turn --frame-stride 10 --with-background --average-tokens --weighting-type mask-error --refine-poses --save-renders --save-metrics
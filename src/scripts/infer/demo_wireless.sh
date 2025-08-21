export CUDA_VISIBLE_DEVICES=2

python demo.py ctdet \
        --dataset table \
        --demo /ssdata/user/gangwei.li/Code/table-tools/datasets/3-31/elect-table-331/labels/val \
        --demo_name demo_wireless \
        --debug 1 \
        --arch dla_34  \
        --K 3000 \
        --MK 5000 \
        --upper_left \
        --tsfm_layers 3\
        --stacking_layers 3 \
        --gpus 0\
        --wiz_2dpe \
        --wiz_detect \
        --wiz_stacking \
        --convert_onnx 0 \
        --vis_thresh_corner 0.6 \
        --vis_thresh 0.5 \
        --scores_thresh 0.3 \
        --nms \
        --load_model /ssdata/user/gangwei.li/Code/LORE-TSR/model/table_elect/model_best.pth \
    	--load_processor /ssdata/user/gangwei.li/Code/LORE-TSR/model/table_elect/processor_best.pth

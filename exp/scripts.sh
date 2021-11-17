# Train on crowdhuman
python train.py mot --arch 'resfpndcn_18' --exp_id crowdhuman_resfpndcn_18_384_224  \
--gpus 0 --batch_size 8 --num_epochs 1 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/crowdhuman.json' \
--input_h 224 --input_w 384 --reid_dim 64 --val_intervals -1

# Train on mix dataset
python train.py mot --arch 'resfpndcn_18' --exp_id mix_resfpndcn_18_384_224 \
 --gpus 0 --batch_size 32 --load_model /home/namtd/workspace/projects/smartcity/src/multiple-tracking/FunMOT/exp/mot/crowdhuman_resfpndcn_18_384_224/model_last.pth \
 --num_epochs 1 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/mix_all.json' --input_h 224 --input_w 384 --reid_dim 64 --val_intervals -1

# Train and finetune on LiveTrack
python train.py mot --arch 'resfpndcn_18' --exp_id finetune_resfpndcn_18_384_224  --gpus 0 --batch_size 32 \
--load_model /home/namtd/workspace/projects/smartcity/src/multiple-tracking/FunMOT/exp/mot/mix_resfpndcn_18_384_224/model_last.pth \
--num_epochs 1 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/mix_track_reids.json' --input_h 224 --input_w 384 --reid_dim 64 --val_intervals -1
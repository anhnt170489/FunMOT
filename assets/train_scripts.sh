# Train on crowdhuman
python train.py mot --arch 'resfpndcn_18' --exp_id livetrack_det_only_resfpndcn_18_570_320  \
--gpus 0,1,2,3 --batch_size 32 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' \
--input_h 320 --input_w 576 --reid_dim 64 --val_intervals -5 --id_weight

# Train on mix dataset
python train.py mot --arch 'resfpndcn_18' --exp_id mix_resfpndcn_18_384_224 \
--gpus 0 --batch_size 32 --load_model /home/namtd/workspace/projects/smartcity/src/multiple-tracking/FunMOT/exp/mot/crowdhuman_resfpndcn_18_384_224/model_last.pth \
--num_epochs 1 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/mix_all.json' --input_h 224 --input_w 384 --reid_dim 64 --val_intervals -1

# Train and finetune on LiveTrack
python train.py mot --arch 'resfpndcn_18' --exp_id finetune_resfpndcn_18_384_224  --gpus 0 --batch_size 32 \
--load_model /home/namtd/workspace/projects/smartcity/src/multiple-tracking/FunMOT/exp/mot/mix_resfpndcn_18_384_224/model_last.pth \
--num_epochs 1 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/mix_track_reids.json' --input_h 224 --input_w 384 --reid_dim 64 --val_intervals -1

# Train on LiveTrack full
python train.py mot --arch 'resfpndcn_18' --exp_id livetrack_full_resfpndcn_18_570_320  \
--gpus 2,3 --batch_size 8 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' \
--input_h 320 --input_w 576 --reid_dim 64 --val_intervals -1

# Train on LiveTrack det only
python train.py mot --arch 'resfpndcn_18' --exp_id livetrack_det_only_resfpndcn_18_570_320  \
--gpus 0,1,2,3 --batch_size 32 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' \
--input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1 --id_weight 0

# One GPU
# Lab
python train.py mot --arch 'resfpndcn_18' --exp_id test_validate  --gpus 7 --batch_size 64 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1 --id_weight 0

# Train detection with pretrained fairmot_dla34 (lab)
python train.py mot --arch 'dla_34' --load_model '../models/FM_pretrained/fairmot_dla34.pth' --exp_id pretrain_dla_det_lab  --gpus 5 --batch_size 64 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1 --id_weight 0

# Train detection with pretrained fairmot_dla34
python train.py mot --arch 'dla_34' --load_model '../models/FM_pretrained/fairmot_dla34.pth' --exp_id pretrain_dla_det  --gpus 6 --batch_size 64 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1 --id_weight 0

# Train detection with pretrained fairmot_dla34 no eval
python train.py mot --arch 'dla_34' --load_model '../models/FM_pretrained/fairmot_dla34.pth' --exp_id pretrain_dla_det_no_eval  --gpus 7 --batch_size 64 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals -1 --id_weight 0

# Train detection with pretrained ctdet_coco_dla_2x
python train.py mot --arch 'dla_34' --load_model '../models/FM_pretrained/fairmot_dla34.pth' --exp_id pretrain_ctdet_coco_dla_2x  --gpus 5 --batch_size 64 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1 --id_weight 0

# Debug train
python train.py mot --arch 'dla_34' --load_model '/home/ubuntu/workspace/namtd/FunMOT/exp/mot/pretrain_ctdet_coco_dla_2x/model_14.pth' --exp_id pretrain_ctdet_coco_dla_2x  --gpus 7 --batch_size 64 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack_debug.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1 --id_weight 0


# Train on pretrained ctdet_coco
python train.py mot --arch 'dla_34' --load_model '../models/FM_pretrained/ctdet_coco_dla_2x.pth' --exp_id ctcoco_pretrained_dla  --gpus 4 --batch_size 64 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1 --id_weight 0
# Train from scrarch based on dla_34
python train.py mot --arch 'dla_34' --exp_id scratch_dla  --gpus 3 --batch_size 64 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1
# Train debug new loss
python train.py mot --arch 'dla_34' --exp_id scratch_dla  --gpus 0 --batch_size 2 --num_epochs 100 --optimizer 'RADAM' --lr 2e-5 --data_cfg '../src/lib/cfg/LiveTrack.json' --input_h 320 --input_w 576 --reid_dim 64 --val_intervals 1
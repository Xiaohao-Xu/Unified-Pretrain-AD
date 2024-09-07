### Pre-training

```bash
# pretraining result saving dir
pretrain_rundir="./runs/ptr-ns_mae-50ep"

# pretraining the embedding network via NS-MAE
torchpack dist-run -np 8  python ./tools/train_ptr.py   ./configs/nuscenes/det/pretrain/ns_mae.yaml   --checkpoint_config.max_keep_ckpts 5  --data.samples_per_gpu 2 --max_epochs 50  --data.train.dataset.load_interval 1 --data.train.times 1   --checkpoint_config.interval 10  --run-dir ${pretrain_rundir} 

# extract the embedding network part which is used for downstream transfer
python ./tools/extract_embed_net.py --run-dir ${pretrain_rundir} 
```

> Notice: As the multi-modal model is partially set in `fp16` mode, there might be instability during training (according to our empirical observations and others). 

> Tip: To reduce the memory cost, you can enable the checkpointing mechanism for the image backbone by setting `with_cp` as `True` (which might slightly enlongate the training time).

### Fine-tuning

#### 3D Object Detection

```bash
# pretraining result saving dir
pretrain_rundir="./runs/ptr-ns_mae-50ep"
# the relative checkpoint path of the pretraining saving dir
transfer_ckpt="/latest_trans_50_e.pth"
# main training for transfer evaluation
transfer_run_dir="./runs/transfer_det-ns_mae-20ep"

torchpack dist-run -np 8 python ./tools/train.py ./configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_del_cbgs.yaml --load_from  "$pretrain_rundir$transfer_ckpt"  --data.train.dataset.load_interval 1 --data.train.times 1  --max_epochs 20  --run-dir ${transfer_run_dir} 

```

#### BEV Map Segmentation

```bash
# pretraining result saving dir
pretrain_rundir="./runs/ptr-ns_mae-50ep"
# the relative checkpoint path of the pretraining saving dir
transfer_ckpt="/latest_trans_50_e.pth"
# main training for transfer evaluation
transfer_run_dir="./runs/transfer_seg-ns_mae-20ep"

torchpack dist-run -np 8 python ./tools/train.py ./configs/nuscenes/seg/fusion-bev256d2-lss_del_cbgs.yaml --load_from  "$pretrain_rundir$transfer_ckpt"  --data.train.dataset.load_interval 1 --data.train.times 1  --max_epochs 20  --run-dir ${transfer_run_dir} 

```

### Label-Efficient Fine-tuning 

#### 3D Object Detection

```bash
# pretraining result saving dir
pretrain_rundir="./runs/ptr-ns_mae-50ep"
# the relative checkpoint path of the pretraining saving dir
transfer_ckpt="/latest_trans_50_e.pth"
# main training for transfer evaluation
transfer_run_dir="./runs/transfer_det-ns_mae-20ep"

# the downsampling ratio during transfer (e.g., ds_ratio=20 indicates 1/20=5% labeled data is used for finetuning)
ds_ratio=20

torchpack dist-run -np 8 python ./tools/train.py ./configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_del_cbgs.yaml --load_from  "$pretrain_rundir$transfer_ckpt"  --data.train.dataset.load_interval ${ds_ratio} --data.train.times ${ds_ratio}  --max_epochs 20  --run-dir ${transfer_run_dir} 

```

#### BEV Map Segmentation

```bash
# pretraining result saving dir
pretrain_rundir="./runs/ptr-ns_mae-50ep"
# the relative checkpoint path of the pretraining saving dir
transfer_ckpt="/latest_trans_50_e.pth"
# main training for transfer evaluation
transfer_run_dir="./runs/transfer_seg-ns_mae-20ep"

# the downsampling ratio during transfer (e.g., ds_ratio=20 indicates 1/20=5% labeled data is used for finetuning)
ds_ratio=20

torchpack dist-run -np 8 python ./tools/train.py ./configs/nuscenes/seg/fusion-bev256d2-lss_del_cbgs.yaml --load_from  "$pretrain_rundir$transfer_ckpt"  --data.train.dataset.load_interval ${ds_ratio} --data.train.times ${ds_ratio}  --max_epochs 20  --run-dir ${transfer_run_dir} 

```


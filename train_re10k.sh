
CUDA_VISIBLE_DEVICES=9 python -m src.main \
    +experiment=re10k \
    checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
    dataset.roots=[/data0/xxy/data/re10k]\
    data_loader.train.batch_size=2 \
    trainer.max_steps=300000 \
    dataset.image_shape=[256,256] \

CUDA_VISIBLE_DEVICES=9 python -m src.main \
    +experiment=re10k \
    dataset.roots=[/data0/xxy/data/re10k]\
    data_loader.train.batch_size=2 \
    trainer.max_steps=300000 \
    dataset.image_shape=[256,256] \
    output_dir='outputs/use_da3'
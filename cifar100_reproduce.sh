# adam
python train.py --optimizer adam --lr 5e-5  
# adam_sn
python train.py --optimizer adamw_sn --lr 1e-3
# adam_snsm
python train.py --optimizer adamw_snsm --lr 1e-3 --rank 64 --update_proj_gap 1000
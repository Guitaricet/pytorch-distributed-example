python -m torch.distributed.launch\
    --nproc_per_node=2\
    --nnodes=2\
    --node_rank=0\
    --master_addr="172.16.33.13"\
    --master_port=1234\
    OUR_TRAINING_SCRIPT.py

python -m torch.distributed.launch\
    --nproc_per_node=2\
    --nnodes=2\
    --node_rank=1\
    --master_addr="172.16.33.13"\
    --master_port=1234\
    OUR_TRAINING_SCRIPT.py

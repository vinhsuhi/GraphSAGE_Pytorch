OUTDIR=visualize/karate
MODEL=graphsage_mean
PREFIX=karate
DS=$HOME/dataspace/graph/${PREFIX}
TRAIN_RATIO=0.8
RATIO=0.1
LR=0.001

############################### compare karate vs karate using vecmap ###############
source activate pytorch

# normal save to original 
python -m graphsage.unsupervised_train --epochs 1000 --model ${MODEL} \
    --prefix ${DS}/graphsage/${PREFIX} \
    --batch_size 8 --print_every 10 \
    --identity_dim 128 \
    --samples_1 8 --samples_2 4 \
    --learning_rate ${LR} \
    --save_embeddings True --base_log_dir ${OUTDIR} \
    --neg_sample_size 2\
    --cuda True \
    --dim_1 128 --dim_2 128





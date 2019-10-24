OUTDIR=visualize/pale_ppi_siamese_clone
MODEL=graphsage_mean
DS=example_data/ppi
DS1=example_data/ppi
PREFIX1=ppi
DS2=example_data/ppi/random_clone
CLONE_RATIO=0.2
PREFIX2=ppi
TRAIN_RATIO=0.2
LR=0.01

############################### compare ppi1 and ppi2 using vecmap ###############
source activate pytorch

python data_utils/random_clone_add.py --input ${DS}/graphsage/ --output ${DS}/random_clone/ --prefix ${PREFIX1} --padd ${CLONE_RATIO} --nadd ${CLONE_RATIO}

# normal save to original 
python -m graphsage.siamese_unsupervised_train --epochs 10 --model ${MODEL} \
    --prefix_source ${DS1}/graphsage/${PREFIX1} \
    --prefix_target ${DS2}/clone,p=${CLONE_RATIO},n=${CLONE_RATIO}/${PREFIX2} \
    --learning_rate ${LR} \
    --save_embeddings True --base_log_dir ${OUTDIR}${CLONE_RATIO} \
    --train_dict_dir ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
    --val_dict_dir ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
    --embedding_loss_weight 1 --mapping_loss_weight 1 \
    --validate_iter 100 \
    --cuda True

source activate vecmap

python vecmap/eval_translation.py ${OUTDIR}/unsup-example_data/graphsage_mean_0.010000/source.emb \
        ${OUTDIR}/unsup-example_data/graphsage_mean_0.010000/target.emb \
        -d ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict --retrieval csls --cuda

###################################################################################
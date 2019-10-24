OUTDIR=visualize/douban
MODEL=graphsage_mean
DS=$HOME/dataspace/graph/flickr_lastfm
DS1=$HOME/dataspace/graph/flickr_lastfm/flickr
PREFIX1=flickr
DS2=$HOME/dataspace/graph/flickr_lastfm/lastfm
PREFIX2=lastfm
TRAIN_RATIO=0.8
LR=0.01

############################### compare ppi1 and ppi2 using vecmap ###############
source activate pytorch

python data_utils/split_dict.py --input ${DS}/dictionaries/groundtruth --out_dir ${DS}/dictionaries/ --split ${TRAIN_RATIO}

python -m graphsage.siamese_unsup_ind_train --cuda True --print_every 100 \
    --prefix_source ${DS1}/graphsage/${PREFIX1} \
    --prefix_target ${DS2}/graphsage/${PREFIX2} \
    --learning_rate ${LR} \
    --embed_epochs 100 --mapping_epochs 100 \
    --train_dict_dir ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
    --val_dict_dir ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
    --identity_dim 128 --validate_iter 100 \
    --base_log_dir ${OUTDIR}/ 

# source activate vecmap

# python vecmap/eval_translation.py ${OUTDIR}/unsup-example_data/graphsage_mean_0.010000/source.emb \
#         ${OUTDIR}/unsup-example_data/graphsage_mean_0.010000/target.emb \
#         -d ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict --retrieval csls --cuda

# ###################################################################################




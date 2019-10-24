# DONE


source activate pytorch

BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/random_clone/del_edge,p=
PREFIX2=ppi
RATIO=0.1
LR=0.01
LOGNAME=sub_vs_sub_del_edge
OUTDIM=150
IDENTITYDIM=150
EMBEDDINGEPOCHS=2000
NEGSAMPLESIZE=10
BATCHSIZEEMBEDDING=512




# RANDOM_CLONE AND SHUFFLE

# python data_utils/random_clone_delete.py --input ${DS1}/graphsage --output ${DS1}/random_clone \
#     --prefix ${PREFIX1} --ndel ${RATIO} --pdel ${RATIO}

# python data_utils/shuffle_graph.py --input_dir ${DS2}${RATIO},n=${RATIO} --out_dir ${DS2}${RATIO},n=${RATIO}/permutation --prefix ${PREFIX2}



# # SPLIT DICT AND RUN MAIN: TRAIN_RATIO 0.2

TRAIN_RATIO=0.2
MAPPINGEPOCHS=2000
BATCHSIZEMAPPING=128

python data_utils/split_dict.py --input ${DS2}${RATIO},n=${RATIO}/permutation/dictionaries/groundtruth --out_dir ${DS2}${RATIO},n=${RATIO}/permutation/dictionaries/ --split ${TRAIN_RATIO}

python -u -m IJCAI16.main_graphsage \
 --prefix1 ${DS1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}${RATIO},n=${RATIO}/permutation/graphsage/${PREFIX2} \
 --train_dict ${DS2}${RATIO},n=${RATIO}/permutation/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS2}${RATIO},n=${RATIO}/permutation/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 ${LR} \
 --out_dim ${OUTDIM} \
 --embedding_epochs ${EMBEDDINGEPOCHS} \
 --identity_dim ${IDENTITYDIM} \
 --mapping_epochs ${MAPPINGEPOCHS} \
 --neg_sample_size ${NEGSAMPLESIZE} \
 --base_log_dir ${BLD} \
 --log_name ${LOGNAME} \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding ${BATCHSIZEEMBEDDING} \
 --batch_size_mapping ${BATCHSIZEMAPPING} \
 --use_features \
 --cuda > logs/${LOGNAME}_${TRAIN_RATIO}


# DS=${DS2}${RATIO},n=${RATIO}/permutation
# EMB_PREFIX=${LOGNAME}_train_percent${TRAIN_RATIO}_GRAPHSAGE
# SOURCE_EMB=${BLD}/embeddings/${EMB_PREFIX}source.emb
# TARGET_EMB=${BLD}/embeddings/${EMB_PREFIX}target.emb


# python -u MUSE/supervised.py \
#     --src_emb ${SOURCE_EMB} \
#     --tgt_emb ${TARGET_EMB} \
#     --emb_dim 300 \
#     --n_refinement 5 \
#     --dico_train ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
#     --dico_eval ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
#     --dico_max_rank 0 > logs/${LOGNAME}_${TRAIN_RATIO}_MUSE





# SPLIT DICT AND RUN MAIN: TRAIN_RATIO 0.03

TRAIN_RATIO=0.03
MAPPINGEPOCHS=100
BATCHSIZEMAPPING=8

# python data_utils/split_dict.py --input ${DS2}${RATIO},n=${RATIO}/permutation/dictionaries/groundtruth --out_dir ${DS2}${RATIO},n=${RATIO}/permutation/dictionaries/ --split ${TRAIN_RATIO}

python -u -m IJCAI16.main_graphsage \
 --prefix1 ${DS1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}${RATIO},n=${RATIO}/permutation/graphsage/${PREFIX2} \
 --train_dict ${DS2}${RATIO},n=${RATIO}/permutation/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS2}${RATIO},n=${RATIO}/permutation/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 ${LR} \
 --out_dim ${OUTDIM} \
 --embedding_epochs ${EMBEDDINGEPOCHS} \
 --identity_dim ${IDENTITYDIM} \
 --mapping_epochs ${MAPPINGEPOCHS} \
 --neg_sample_size ${NEGSAMPLESIZE} \
 --base_log_dir ${BLD} \
 --log_name ${LOGNAME} \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding ${BATCHSIZEEMBEDDING} \
 --batch_size_mapping ${BATCHSIZEMAPPING} \
 --cuda > logs/${LOGNAME}_${TRAIN_RATIO}


# DS=${DS2}${RATIO},n=${RATIO}/permutation
# EMB_PREFIX=${LOGNAME}_train_percent${TRAIN_RATIO}_GRAPHSAGE
# SOURCE_EMB=${BLD}/embeddings/${EMB_PREFIX}source.emb
# TARGET_EMB=${BLD}/embeddings/${EMB_PREFIX}target.emb


# python -u MUSE/supervised.py \
#     --src_emb ${SOURCE_EMB} \
#     --tgt_emb ${TARGET_EMB} \
#     --emb_dim 300 \
#     --n_refinement 5 \
#     --dico_train ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
#     --dico_eval ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
#     --dico_max_rank 0 > logs/${LOGNAME}_${TRAIN_RATIO}_MUSE


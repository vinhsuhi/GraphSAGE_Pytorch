# DONE


source activate pytorch

BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/random_delete_node/del,p=
PREFIX2=ppi
RATIO=0.2
LR=0.01
LOGNAME=sub_vs_sub_del_nodes02
OUTDIM=300
EMBEDDINGEPOCHS=2000
NEGSAMPLESIZE=10
BATCHSIZEEMBEDDING=512



TRAIN_RATIO=0.2
MAPPINGEPOCHS=2000
BATCHSIZEMAPPING=128


python -u -m IJCAI16.main \
 --prefix1 ${DS1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}${RATIO}/permutation/graphsage/${PREFIX2} \
 --train_dict ${DS2}${RATIO}/permutation/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS2}${RATIO}/permutation/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 ${LR} \
 --embedding_dim ${OUTDIM} \
 --embedding_epochs ${EMBEDDINGEPOCHS} \
 --mapping_epochs ${MAPPINGEPOCHS} \
 --neg_sample_size ${NEGSAMPLESIZE} \
 --base_log_dir ${BLD} \
 --log_name ${LOGNAME}_train_percent${TRAIN_RATIO} \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding ${BATCHSIZEEMBEDDING} \
 --batch_size_mapping ${BATCHSIZEMAPPING} \
 --cuda > logs/${LOGNAME}_${TRAIN_RATIO}_IJCAI16



TRAIN_RATIO=0.03
MAPPINGEPOCHS=100
BATCHSIZEMAPPING=8



python -u -m IJCAI16.main \
 --prefix1 ${DS1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}${RATIO}/permutation/graphsage/${PREFIX2} \
 --train_dict ${DS2}${RATIO}/permutation/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS2}${RATIO}/permutation/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 ${LR} \
 --embedding_dim ${OUTDIM} \
 --embedding_epochs ${EMBEDDINGEPOCHS} \
 --mapping_epochs ${MAPPINGEPOCHS} \
 --neg_sample_size ${NEGSAMPLESIZE} \
 --base_log_dir ${BLD} \
 --log_name ${LOGNAME}_train_percent${TRAIN_RATIO} \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding ${BATCHSIZEEMBEDDING} \
 --batch_size_mapping ${BATCHSIZEMAPPING} \
 --cuda > logs/${LOGNAME}_${TRAIN_RATIO}_IJCAI16
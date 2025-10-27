#!/bin/bash

mkdir -p age_predict_with_imput_depth

for ds in {RB_GALAXY,RB_GDNA_GALAXY,RB_GDNA_TWIST,RB_TWIST}; do
	python3 age_predict_with_imput_depth.py \
		-b data/$ds/beta.tsv \
		-d data/$ds/depth.tsv \
		-m data/$ds/metadata.tsv \
		-C dnamphenoage,pcphenoage,grimage,pcgrimage,grimage2,han,hannum,pchannum,horvath2013,lin,pchorvath2013,skinandblood,pcskinandblood,pipekfilteredh,pipekretrainedh,stoch,stocp,stocz,zhangblup,zhangen \
		-I mean,median,constant,knn \
		-D 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
		-o age_predict_with_imput_depth/$ds.pkl
done
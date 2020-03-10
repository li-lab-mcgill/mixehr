#!/bin/bash

ehrdata=./mixmimic/mimic_trainData.txt
ehrmeta=./mixmimic/mimic_meta.txt

# infer metaphe of all training pat
testdata=$ehrdata

K=75
# K=50
niter=500


outdir=./mixmimic

trainedPrefix=$outdir/mimic_trainData_JCVB0_nmar_K${K}_iter$niter

mixehr -m $ehrmeta -n JCVB0 --newPatsData $testdata \
	--trainedModelPrefix $trainedPrefix -k $K --inferNewPatentMetaphe \
	--inferPatParams_maxiter 100


#!/bin/bash

ehrdata=./mixmimic/mimic_trainData.txt
ehrmeta=./mixmimic/mimic_meta.txt

K=75
# K=50

niter=500

# mar model (evaluation purpose only)
# ./mixehr --outputIntermediates -f $ehrdata -m $ehrmeta -k $K -i $niter --inferenceMethod JCVB0 --mar

# nmar model
./mixehr --outputIntermediates -f $ehrdata -m $ehrmeta -k $K -i $niter --inferenceMethod JCVB0 

# stochastic variational inference for massive scale EHR
# bathsize=5000
# stepsize=0.6
# burnin=10
# ./mixehr --outputIntermediates -f $ehrdata -m $ehrmeta -b $batchsize -k $K -i $niter -n SJCVB0 -r $burnin -s $stepsize


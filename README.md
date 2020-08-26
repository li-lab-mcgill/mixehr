# MixEHR: Multi-modal Mixture Topic Model for mining EHR data

 
![mixehr](images/mixehr_overview.png)
**MixEHR model overview**. **a.** Multi-view matrix factorization of multiple data matrices corresponding to different EHR data types including lab tests, billing code, doctor notes, etc. **b.** Proposed Bayesian model for modeling non-missing at random (NMAR) lab tests and other multimodal data. In order to achieve tractable inference, we assign a latent topic ![formula](https://render.githubusercontent.com/render/math?math=h_{lj}) to the lab results ![formula](https://render.githubusercontent.com/render/math?math=y_{lj}) and missing indicator (![formula](https://render.githubusercontent.com/render/math?math=r_{lj})) such that they become conditionally independent. **c.** Collapsed variational Bayesian inference of the MixEHR model. The inference and learning can be visualized as marginalizing a 3-dimensional tensor that represents the expectations of the latent variables.


MixEHR is a Unix-style command-line tool. You can compile it on a unix machine. 

## INSTALLATION:

To install MixEHR, you will need to first install armadillo (http://arma.sourceforge.net)

Assuming you are in the mixehr directory, to compile, simply run:
```
make
```

To test whether ‘mixehr’ can run, do: 
```
mixehr -h
```
This should output:
```
./mixehr -f examples/toydata.txt -m examples/toymeta_phe.txt -i 10 -k 10
```

See scripts [mixmimic.sh](mixmimic.sh) for training on MIMIC data

We included the MIMIC-III data. To run mixehr, you will need a meta data file that contains 3 columns: 

1. typeId: indicates distinct data types such as ICD-9, lab test, etc with 1, 2, etc
2. pheId: indicate the phenotype ID (e.g., lab test 1, lab test 2, etc)
3. stateCnt: indicate the number of states for the phenotype. This is designed for lab test at the moment, but will work for other data types with discrete states.

See the example file [mixmimic/mimic_meta.txt.gz](mixmimic/mimic_meta.txt.gz) in the folder.

The actual EHR data file [mixmimic/mimic_trainData.txt.gz](mixmimic/mimic_trainData.txt.gz) has 5 columns rows:

1. Patient ID
2. typeId (concurring the meta file above)
3. pheId (concurring the meta file above)
4. stateId (zero-based, and set to 1 for binary feature and starting 0 onwards for discrete lab values)
5. freq (number of times observed at least 1)

NOTE: stateId must be incremental and start from 0. That is no skipping number. 

## Training and validation
The main training command:
```
./mixehr -f $ehrdata -m $ehrmeta -k $K -i $niter \
	--inferenceMethod JCVB0 --maxcores 8 \
	--outputIntermediates 
```

Flags are:
```
-f: ehr data file 
-m: meta file 
-i: number of iterations 
-k: number of meta-phenotypes 
-n: inference method (JCVB0 or SCVB0 for stochastic) 
--maxcores: maximum number of CPU cores to use 
--outputIntermediates: (whether output intermediate learned parameters for inspection)  
```

If you have a test data for producing predictive likelihood, then you can run the same command with added flag `-t $testdata`, where the test data contain the same format as the training data but contain one extra column in the end indicating whether the feature is missing (0) or observed (1). See [examples](examples) folder for the simulated data as examples.


## Infer new patient mixture
See [mixmimic_testpat.sh](mixmimic_testpat.sh)

After you train the model, you will find a CSV file `mimic_trainData_JCVB0_iter500_phi_normalized`. The first two columns are typeId and pheId (concurring the IDs in the above meta file [mixmimic/mimic_meta.txt](mixmimic/mimic_meta.txt)). The rest of the columns are normalized probabilities for the membership of each phenotype. Similar phenotypes tend to exhibit high probabilities under the same column (i.e., meta-phenotype). `mimic_trainData_JCVB0_iter500_eta_normalized` is similar but with the first 3 columns, indicating typeId, labId, stateId and the rest are K columns probabilities.


Command to infer disease mixture of new patients:
```
mixehr -m $ehrmeta -n JCVB0 --newPatsData $testdata \
        --trainedModelPrefix $trainedPrefix -k $K --inferNewPatentMetaphe \
        --inferPatParams_maxiter 100
```
This gives a D by K matrix file (*_metaphe.csv), which is the normalized probabilities (row sum is one) for D test patients for K meta-phenotypes.

These inferred disease mixtures can then be used as patient representations to train classifiers for specific tasks. For example, a linear classifier (such as Logistic Regression or Elastic Net) can be used to predict mortality given these patient representations as input. 

## Application 1: Prediction of mortality using patient topic mixture memberships:
One way to interpret the mortality prediction results and the topic mixtures is to calculate the correlation between the mortaltiy labels and each of the topics. The topics most positively and negatively correlated with mortality can be visualized as heat maps. To do so, we make use of the file with the suffix  `_phi_normalised.csv` that is obtained after training MixEHR. This file contains an N by K matrix file which is the normalized probabilities (column sum is one) for N features for K meta-phenotypes (topics). For each of the K topics, the top features can be obtained from the feature IDs given in [mixmimic/ehrFeatId.Rdata](mixmimic/ehrFeatId.RData). 

A heatmap of these features can then be plotted whose intensity is given by the probability values. Below is an example of the top 5 EHR codes (features) associated with the top 3 topics positively correlated and the bottom 3 topics negatively correlated with mortality indicated by blue and green respectively. The categories of the features are indicated by the colour map presented beside the heatmap. The intensity of the red colour indicates the probability of a particular feature belonging to a particular topic. 

![heatmap](images/heatmap.png)


## Application 2: Retrospective Prediction of EHR code:
![retro_code_prediction](images/ehr_code_prediction.png)

To impute missing data in an individual-specific way, we here describe a k-nearest neighbour approach. The prediction can be divided into 3 steps:

1. Train MixEHR on training set to learn the EHR-code by disease topic matrices **W** across data types and infer the disease topic mixtures ![formula](https://render.githubusercontent.com/render/math?math=\theta^{train}) for each training patient data point;

2. To infer the probability of an unknown EHR code `t` for a test patient `j'`, use MixEHR and the learnt disease topic matrices **W** to infer the disease topic mixture <img src="https://render.githubusercontent.com/render/math?math=\theta_{j'}"> for the test patient;

3. Compare the test patient disease topic mixture <img src="https://render.githubusercontent.com/render/math?math=\theta_{j'}"> with the training patient disease mixtures <img src="https://render.githubusercontent.com/render/math?math=\theta^{train}"> to find the `k` most similar training patients <img src="https://render.githubusercontent.com/render/math?math=\mathcal{S}_{j'}">. Here the patient-patient similarity matrix is calculated based on the Euclidean distance between their disease topic mixtures:

Finally, we take the average of the EHR code t over these k-nearest neighbour patients as the prediction for the target code t for test patient `j'`. We empirically determined the number of nearest neighbours `k` to be 100.


Please download and unzip this file: [ehr_code_pred.tar.gz](https://drive.google.com/file/d/1i6FxKSOaagiR0MXMhiHkQtIoyab3eBVE/view?usp=sharing)

Then run the following command:

```
./mixehr --metaFile metainfo.txt --topics 75 \ 
	--trainDataFile train${i}.txt \
	--imputeTargetsFile impute_target_pheId.txt \
	--imputePatDataFile test_data.txt \
	--knn_impute 25 \
	--output_dir impute_data \
	--trainedModelPrefix train${i}_JCVB0_nmar_K75_iter200
```

The predictions are saved in files `target_phe_pred.csv` under directory `impute_data`. Rows are admissions or patients and columns are EHR code.

## Application 3: Imputing missing lab results:
![lab_imputation](images/lab_imputation.png)


**Workflow to impute lab results.** 
This is similar to the retrospective EHR code prediction. 
- **Step 1.** We modeled lab tests, lab test results and non-lab EHR data (i.e., ICD, notes, prescription, treatment) to infer the patient topic mixture. 
- **Step 2.** For a test patient, we masked each of his observed lab test result t and inferred his topic mixture. 
- **Step 3.** We then found k=25 (by default) patients who have the lab test results `t` observed and exhibit the most similar topic mixture to the test patient. We then took the average of lab result values over the `k` patients as the prediction of the lab result value for the test patient `j'`. Steps 1-3 were repeated to evaluate every observed lab test in every test patient.


Please download and unzip this file: [lab_imputation.tar.gz](https://drive.google.com/file/d/1q9O8WL4kkG0fDv_6Ootjs6Q59sD7A1EP/view?usp=sharing)


Here the `mimic_data_train_pat_mix_50.csv` that can be generated from using above approach described in **Infer new patient mixture**. The `mimic_data_train_JCVB0_nmar_K50_iter1000*.csv` are the trained model parameters that can be generated from **Training and validation** section.

Then run the following command:

```
k=50
iter=1000

./mixehr --metaFile lab_imputation/mimic_meta.txt --topics $k \
        --trainDataFile lab_imputation/mimic_data_train.txt \
        --imputeTargetsFile lab_imputation/impute_target_labId.txt \
        --imputePatDataFile lab_imputation/mimic_data_test.txt \
        --trainPatMetapheFile lab_imputation/mimic_data_train_pat_mix_${k}.csv \
        --trainPatIdFile lab_imputation/mimic_data_train_pat_mix_${k}_patId.csv \
        --knn_impute 25 \
        --output_dir lab_imputation/mimic_data_test_pat_impute_lab_K${k} \
        --trainedModelPrefix lab_imputation/mimic_data_train_JCVB0_nmar_K${k}_iter$iter
```        

The predictions are saved in files `target_lab_res_pred.csv` under directory `mimic_data_test_pat_impute_lab_K50`. Rows are admissions or patients and columns are lab tests. Here we predict binary results (normal and abnormal). Therefore, we only need to save the predictions for abnormal (i.e., one column per lab test). It is also easy to predict lab test with more than two result values (e.g., low, normal, high).

## Application 4: Prediction of longitudinal EHR code:
![code_prediction](images/code_prediction.png)

To predict dynamic or longitudinal EHR code, we describe a novel pipeline that combines MixEHR topics with recurrent neural network (RNN) with Gated Recurrent Unit (GRU). We first trained MixEHR on the EHR data for 39,000 patients with single-admission in MIMIC-III. We then used the trained MixEHR to infer topic mixture at each admission for the 7541 patients with multiple admissions. Then we used as input the inferred topic mixture at the current admission (say at time `t`) to the RNN to autoregressively predict the diagnostic codes at the next admission at time `t+1`. Here MixEHR uses all of the data types from MIMIC-III. More details on the architecture of the neural networks are described in our paper (under peer review). The lines of code given below may be followed to use the inferred MixEHR mixtures for longitudinal EHR code prediction using an RNN.

After training MixEHR for `k` topics and then inferring the topic mixtures for some test data, this test data can be further split into train (for RNN) and test (for RNN). Since we wish to do longitudinal EHR code prediction, we need to obtain the topic mixtures for all admissions for each patient. In order to do so, we infer the topic mixtures for each admission. This topic mixture needs to then be merged with the patient_ids. They can then be grouped by SUBJECT_ID. This will give us a list of list of topic mixtures where the outer list is for patients and the inner list contains the mixture probabilities for the admissions for that patient. Below is an example of 75 topic mixtures for 1 patient with 2 admissions. This array has 2 lists (2 admissions) and each of the inner lists has 75 values which indiciate probabilities of belonging to the 75 topics. 

```
array([[0.01209019, 0.01998611, 0.01149166, 0.01368793, 0.01413315,
        0.01054217, 0.01356388, 0.00621338, 0.01544446, 0.00709538,
        0.01620278, 0.00614183, 0.00771023, 0.0090849 , 0.01131411,
        0.01241052, 0.0144826 , 0.0146653 , 0.02000644, 0.01497399,
        0.01340695, 0.01350807, 0.00276752, 0.01641031, 0.01669823,
        0.01951223, 0.01066794, 0.01707473, 0.0112766 , 0.01031544,
        0.01553755, 0.0104024 , 0.01191917, 0.00961406, 0.00993082,
        0.00500289, 0.01012674, 0.00821885, 0.012427  , 0.01166625,
        0.00727486, 0.01105222, 0.00953794, 0.02064406, 0.01850404,
        0.01508257, 0.02231016, 0.01588202, 0.01961249, 0.01600174,
        0.01701199, 0.0125113 , 0.01029791, 0.01124358, 0.01373042,
        0.01646089, 0.01997158, 0.01029284, 0.01409426, 0.01987322,
        0.00772018, 0.00979768, 0.00951799, 0.01881881, 0.01652693,
        0.01186132, 0.0204533 , 0.02348759, 0.00991412, 0.01767057,
        0.00685997, 0.01342949, 0.01891696, 0.019939  , 0.00597127],
       [0.00837015, 0.01197824, 0.00547403, 0.00485901, 0.00421585,
        0.01208326, 0.01063464, 0.0056178 , 0.0100289 , 0.00625341,
        0.00902622, 0.00731984, 0.01540566, 0.00863343, 0.00705029,
        0.00974445, 0.00965424, 0.00510858, 0.01167708, 0.01177612,
        0.00791745, 0.00481467, 0.00108998, 0.01148122, 0.01069184,
        0.0052892 , 0.01377428, 0.0109204 , 0.00681426, 0.01312267,
        0.01124336, 0.00797052, 0.00507396, 0.0130879 , 0.00626762,
        0.00302014, 0.006392  , 0.01025577, 0.01068186, 0.01471757,
        0.00883061, 0.00896801, 0.01005405, 0.01116567, 0.01019837,
        0.0084849 , 0.01274504, 0.00740643, 0.00868885, 0.01151242,
        0.00630296, 0.01335733, 0.01524496, 0.00822703, 0.00809452,
        0.00584196, 0.00530626, 0.00624914, 0.00632897, 0.24853322,
        0.00455988, 0.00873241, 0.00783489, 0.00897402, 0.01274341,
        0.00594328, 0.01528712, 0.08298595, 0.00868986, 0.00975242,
        0.00825348, 0.0093502 , 0.00620293, 0.02367582, 0.01593576]])
```

Since we have the patient_ids, the corresponding labels (diagnosis codes) can be obtained from the `ICD_DIAGNOSIS.csv` file in MIMIC-III. This list of lists can then be split into `train_set_x` and `test_set_x`. The corresponding labels are split into `train_set_y` and `test_set_y`. Since patients have different number of admissions, these datasets are then padded for uniformity. The processed files are available as pickled files in the data download link.

`num_classes` is the total number of unique ICD codes available in MIMIC-III dataset.

```
import pickle as pkl
from keras.models import Sequential
from keras.layers import GRU, Dense, Activation

train_set_x = pkl.load(open("/path/to/downloaded/train_set_x.pkl","rb"))
train_set_y = pkl.load(open("/path/to/downloaded/train_set_y.pkl","rb"))

test_set_x = pkl.load(open("/path/to/downloaded/test_set_x.pkl","rb"))
test_set_y = pkl.load(open("/path/to/downloaded/test_set_y.pkl","rb"))

batchsize = 64
n_batches = int(np.ceil(float(len(train_set_x)) / float(batchsize)))
max_epochs = 70

# RNN code - 2 layer GRU
model = Sequential()
model.add(GRU(128, input_shape=(len(test_set_X[0]), 75), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256, input_shape=(128, 75), return_sequences=True))
model.add(Dense(num_classes, input_shape=(256,)))
model.add(Activation('sigmoid'))

sgd = opt.SGD(lr=0.1, decay=1e-6, momentum=0., nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X, y, batch_size=batchsize, epochs=max_epochs, verbose=2, validation_split=0.2)

# plot loss during training
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()

# obtain the prediction probabilities of the model
pred_test = model.predict_proba(x_test)
```
The obtained predicted probabilities (`pred_test`) can be used to measure accuracy against `test_set_y` which are the true labels. 

# Training

Train a new model is separated into three main parts:

- [Preprocessing: Subsample one sample](#preprocessing-subsample-one-sample)
- [Build bin for training](#build-bin-for-training)
  - [Option 1. single sample](#option-1-single-sample)
  - [Option 2. multiple samples](#option-2-multiple-samples)
- [Train a new model](#train-a-new-model)

---

## Preprocessing: Subsample one sample

If targeted to build a training dataset with multiple subsamples, before the bin building process, the original BAM file should subsampled into multiple BAM files first.

```bash
# make sure the provided bam file is sorted and indexed (e.g. hg001.bam)
BAM_FILE_PATH="[YOUR_BAM_FILE_PATH]"

# make sure the folder exists
SUBSAMPLED_BAMS_FOLDER_PATH="[SUBSAMPLED_BAMS_FOLDER_PATH]"

# FRAC values for samtools view -s INT.FRAC
# check samtools view -s documentation for further information
DEPTHS=(800 400 200 100)

THREADS=24

# subsample
for i in "${!DEPTHS[@]}"
do
  samtools view -@ ${THREADS} -s ${i}.${DEPTHS[i]} -b ${BAM_FILE_PATH} \
  > ${SUBSAMPLED_BAMS_FOLDER_PATH}/0.${DEPTHS[i]}.bam
  samtools index -@ ${THREADS} ${SUBSAMPLED_BAMS_FOLDER_PATH}/0.${DEPTHS[i]}.bam
done

# add symbolic link named 1.000.bam and 1.000.bam.bai and use it later
ln -s ${BAM_FILE_PATH} ${SUBSAMPLED_BAMS_FOLDER_PATH}/1.000.bam
ln -s ${BAM_FILE_PATH}.bai ${SUBSAMPLED_BAMS_FOLDER_PATH}/1.000.bam.bai
```

## Build bin for training

### Option 1. Single Sample

This option provides a building bin script for one sample. (Included both single sample without subsample or single sample with multiple subsamples) \
> - Intended to separate the script into many script-block for better understanding the whole process to generate a training dataset.
Moreover, It is easier to trace errors if run the script block-by-block.
> - for each `parallel` command with `--joblog` option, we can check `Exitval` column from the job log output. If the column contains non-zero value, you may try to re-run the script-block again.
> - Absolute path is always preferred when using this script.

#### 1. Setup variables for building bin
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"                               # e.g. ./clair.py
PYPY="[PYPY_BIN_PATH]"                                         # e.g. pypy3

VCF_FILE_PATH="[YOUR_VCF_FILE_PATH]"                           # e.g. hg001.vcf.gz
BAM_FILE_PATH="[YOUR_BAM_FILE_PATH]"                           # e.g. hg001.bam
REFERENCE_FILE_PATH="[YOUR_FASTA_FILE_PATH]"                   # e.g. hg001.fasta

# dataset output folder (the directory will be created later)
DATASET_FOLDER_PATH="[OUTPUT_DATASET_FOLDER_PATH]"

# subsamples array, (1.000) for single sample without subsample
DEPTHS=(1.000 0.800)

# expected to have bams named in DEPTHS array (e.g. 1.000.bam 0.800.bam)
# check `Preprocessing: Subsample one sample` section for further information
SUBSAMPLED_BAMS_FOLDER_PATH="[SUBSAMPLED_BAMS_FOLDER_PATH]"

# chromosomes prefix ("chr" if chromosome name have "chr"-prefix)
CHR_PREFIX=""

# chromosomes array (no need to include any "chr"-prefix)
CHR=(21 22 X)

# no of threads
THREADS=24

# for some memory intensive options, may use this value instead of THREADS
THREADS_LOW=10

DEPTHS_PER_SAMPLE=${#DEPTHS[@]}
ESTIMATED_SPLIT_NO_OF_LINES=$((180000 * $DEPTHS_PER_SAMPLE))
MINIMUM_COVERAGE=4

VARIANT_FOLDER_PATH="${DATASET_FOLDER_PATH}/var"
CANDIDATE_FOLDER_PATH="${DATASET_FOLDER_PATH}/can"
TENSOR_VARIANT_FOLDER_PATH="${DATASET_FOLDER_PATH}/tensor_var"
TENSOR_CANDIDATE_FOLDER_PATH="${DATASET_FOLDER_PATH}/tensor_can"
TENSOR_PAIR_FOLDER_PATH="${DATASET_FOLDER_PATH}/tensor_pair"
SHUFFLED_TENSORS_FOLDER_PATH="${DATASET_FOLDER_PATH}/all_shuffled_tensors"
BINS_FOLDER_PATH="${DATASET_FOLDER_PATH}/all_bins"
```

#### 2. Create Directories
```bash
mkdir ${DATASET_FOLDER_PATH}
cd ${DATASET_FOLDER_PATH}
mkdir ${VARIANT_FOLDER_PATH}
mkdir ${CANDIDATE_FOLDER_PATH}
mkdir ${TENSOR_VARIANT_FOLDER_PATH}
mkdir ${TENSOR_CANDIDATE_FOLDER_PATH}
mkdir ${TENSOR_PAIR_FOLDER_PATH}
mkdir ${SHUFFLED_TENSORS_FOLDER_PATH}
mkdir ${BINS_FOLDER_PATH}

# create directories for different depths
for j in "${!DEPTHS[@]}"
do
  cd ${TENSOR_VARIANT_FOLDER_PATH}
  mkdir ${DEPTHS[j]}

  cd ${TENSOR_CANDIDATE_FOLDER_PATH}
  mkdir ${DEPTHS[j]}

  cd ${TENSOR_PAIR_FOLDER_PATH}
  mkdir ${DEPTHS[j]}
done

cd ${DATASET_FOLDER_PATH}
```

#### 3. Get variant information using `GetTruth` submodule
```bash
parallel --joblog ./get_truth.log -j${THREADS} \
"${PYPY} ${CLAIR} GetTruth \
--vcf_fn ${VCF_FILE_PATH} \
--var_fn ${VARIANT_FOLDER_PATH}/var_{1} \
--ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]}

# merge all variants into a single file (named all_var)
cat ${VARIANT_FOLDER_PATH}/var_* > ${VARIANT_FOLDER_PATH}/all_var
```

#### 4. Get information from random positions (as candidates) using `ExtractVariantCandidates` submodule
```bash
parallel --joblog ./evc.log -j${THREADS} \
"${PYPY} ${CLAIR} ExtractVariantCandidates \
--bam_fn ${BAM_FILE_PATH} \
--ref_fn ${REFERENCE_FILE_PATH} \
--can_fn ${CANDIDATE_FOLDER_PATH}/can_{1} \
--ctgName ${CHR_PREFIX}{1} \
--gen4Training" ::: ${CHR[@]}
```

#### 5. Create tensors with variant information using `CreateTensor` submodule
```bash
parallel --joblog ./create_tensor_var.log -j${THREADS} \
"${PYPY} ${CLAIR} CreateTensor \
--bam_fn ${SUBSAMPLED_BAMS_FOLDER_PATH}/{1}.bam \
--ref_fn ${REFERENCE_FILE_PATH} \
--can_fn ${VARIANT_FOLDER_PATH}/var_{2} \
--minCoverage ${MINIMUM_COVERAGE} \
--tensor_fn ${TENSOR_VARIANT_FOLDER_PATH}/{1}/tensor_var_{2} \
--ctgName ${CHR_PREFIX}{2}" ::: ${DEPTHS[@]} ::: ${CHR[@]}
```

#### 6. Create tensors with candidate infomation using `CreateTensor` submodule
```bash
parallel --joblog ./create_tensor_can.log -j${THREADS} \
"${PYPY} ${CLAIR} CreateTensor \
--bam_fn ${SUBSAMPLED_BAMS_FOLDER_PATH}/{1}.bam \
--ref_fn ${REFERENCE_FILE_PATH} \
--can_fn ${CANDIDATE_FOLDER_PATH}/can_{2} \
--minCoverage ${MINIMUM_COVERAGE} \
--tensor_fn ${TENSOR_CANDIDATE_FOLDER_PATH}/{1}/tensor_can_{2} \
--ctgName ${CHR_PREFIX}{2}" ::: ${DEPTHS[@]} ::: ${CHR[@]}
```
> - If you have plenty amount of computing resources, run step 5 and 6 in parallel to speed up the process.

#### 7. Create tensor pair using `PairWithNonVariants` submodule
```bash
parallel --joblog ./create_tensor_pair.log -j${THREADS} \
"${PYPY} ${CLAIR} PairWithNonVariants \
--tensor_can_fn ${TENSOR_CANDIDATE_FOLDER_PATH}/{1}/tensor_can_{2} \
--tensor_var_fn ${TENSOR_VARIANT_FOLDER_PATH}/{1}/tensor_var_{2} \
--output_fn ${TENSOR_PAIR_FOLDER_PATH}/{1}/tensor_pair_{2} \
--amp 2" ::: ${DEPTHS[@]} ::: ${CHR[@]}
```

#### 8. Shuffle, split and compress shuffled tensors
```bash
# first round shuffle
ls tensor_pair/*/tensor_pair* | \
parallel --joblog ./uncompress_tensors_round_1.log -j${THREADS} \
--line-buffer --shuf --verbose --compress gzip -dc ::: | \
parallel --joblog ./round_robin_cat_round_1.log -j${THREADS} \
--line-buffer --pipe -N1000 --no-keep-order --round-robin --compress cat | \
split -l ${ESTIMATED_SPLIT_NO_OF_LINES} \
--filter='shuf | pigz > $FILE.gz' -d - ${SHUFFLED_TENSORS_FILE_PATH}/round1_

# second round shuffle
ls ${SHUFFLED_TENSORS_FILE_PATH}/round1_* | \
parallel --joblog ./uncompress_tensors.log -j${THREADS_LOW} \
--line-buffer --shuf --verbose --compress gzip -dc ::: | \
parallel --joblog ./round_robin_cat.log -j${THREADS} \
--line-buffer --pipe -N1000 --no-keep-order --round-robin --compress cat | \
split -l ${ESTIMATED_SPLIT_NO_OF_LINES} \
--filter='shuf | pigz > $FILE.gz' -d - ${SHUFFLED_TENSORS_FILE_PATH}/split_
```
> - If shuffle is not very important to you, you may consider not to shuffle or shuffle one round only
> - if have enough memory resources, welcome to increase # of threads for faster shuffling process.

#### 9. Create small bins for each shuffled tensors using `Tensor2Bin` submodule
```bash
ls ${SHUFFLED_TENSORS_FOLDER_PATH}/split_* | \
parallel --joblog ./tensor2Bin.log -j${THREADS_LOW} \
"python ${CLAIR} Tensor2Bin \
--tensor_fn {} \
--var_fn ${VARIANT_FOLDER_PATH}/all_var \
--bin_fn ${BINS_FOLDER_PATH}/{/.}.bin \
--allow_duplicate_chr_pos"
```

#### 10. Combine small bins into a training dataset using `CombineBins` submodule
```bash
python ${CLAIR} CombineBins
```

---

### Option 2. multiple samples

This option provides a script for building a bin with multiple simples. \
Like Option 1, it is intended to separate the script into many script-block for better understanding the whole process to generate a training dataset.
Moreover, It is easier to trace errors if run the script block-by-block.
> - for each `parallel` command with `--joblog` option, we can check `Exitval` column from the job log output. If the column contains non-zero value, you may try to re-run the script-block again.
> - Absolute path is always preferred when using this script.
#### 0. For each sample, apply [Option 1](#option-1-single-sample) from step 1 to step 7. <br> Make sure  DATASET_FOLDER_PATH are different for different samples.

#### 1. Setup variables for building bin
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"                               # e.g. ./clair.py
PYPY="[PYPY_BIN_PATH]"                                         # e.g. pypy3

# dataset output folder (the directory will be created later)
TARGET_DATASET_FOLDER_PATH="[TARGET_DATASET_FOLDER_PATH]"

# one line for one sample
SOURCE_SAMPLE_NAMES=(
  "hg001"
  "hg002"
)
# provide DATASET_FOLDER_PATH(s) in Option 1.
SOURCE_DATASET_PATHS=(
  "[DATASET_FOLDER_PATH_FOR_SAMPLE_1]"
  "[DATASET_FOLDER_PATH_FOR_SAMPLE_2]"
)
# make sure the prefixes are different for different samples
SOURCE_SAMPLE_PREFIX=(
  "u"
  "v"
)

SAMPLE_1_DEPTHS=(1.000)
SAMPLE_1_CHR=(21 22 X)

SAMPLE_2_DEPTHS=(1.000)
SAMPLE_2_CHR=(21 22)

ALL_SAMPLES_SOURCE_DEPTHS=(
  SAMPLE_1_DEPTHS[@]
  SAMPLE_2_DEPTHS[@]
)
ALL_SAMPLES_SOURCE_CHR=(
  SAMPLE_1_CHR[@]
  SAMPLE_2_CHR[@]
)

# chromosomes prefix ("chr" if chromosome name have "chr"-prefix)
CHR_PREFIX=""

NO_OF_SAMPLES=${#ALL_SAMPLES_SOURCE_DEPTHS[@]}
DEPTHS_PER_SAMPLE=${#SAMPLE_1_DEPTHS[@]}
ESTIMATED_SPLIT_NO_OF_LINES=$((90000 * $DEPTHS_PER_SAMPLE * $NO_OF_SAMPLES))

# no of threads
THREADS=24

# for some memory intensive options, may use this value instead of THREADS
THREADS_LOW=10
```

#### 2. Create directories
```bash
VARIANT_FOLDER_PATH="${TARGET_DATASET_FOLDER_PATH}/var"
TENSOR_PAIR_FOLDER_PATH="${TARGET_DATASET_FOLDER_PATH}/tensor_pair"
SHUFFLED_TENSORS_FOLDER_PATH="${TARGET_DATASET_FOLDER_PATH}/all_shuffled_tensors"
BINS_FOLDER_PATH="${TARGET_DATASET_FOLDER_PATH}/all_bins"

mkdir ${TARGET_DATASET_FOLDER_PATH}
cd ${TARGET_DATASET_FOLDER_PATH}
mkdir ${VARIANT_FOLDER_PATH}
mkdir ${TENSOR_PAIR_FOLDER_PATH}
mkdir ${SHUFFLED_TENSORS_FOLDER_PATH}
mkdir ${BINS_FOLDER_PATH}
```

#### 3. Create symbolic links for variant information from different samples
```bash
for s in "${!SOURCE_SAMPLE_NAMES[@]}"
do
  SAMPLE_NAME=${SOURCE_SAMPLE_NAMES[s]}
  echo "[INFO] Get Truth for sample ${SAMPLE_NAME}"

  SAMPLE_DIR="${VARIANT_FOLDER_PATH}/${SAMPLE_NAME}"
  mkdir ${SAMPLE_DIR}

  SAMPLE_SOURCE_CHR=(${!ALL_SAMPLES_SOURCE_CHR[s]})
  for i in "${!SAMPLE_SOURCE_CHR[@]}"
  do
    ln -s \
    ${SOURCE_DATASET_PATHS[s]}/var/var_${SAMPLE_SOURCE_CHR[i]} \
    ${VARIANT_FOLDER_PATH}/${SAMPLE_NAME}/var_${SAMPLE_SOURCE_CHR[i]}
  done

  cd ${SAMPLE_DIR}
  cat var_* > all_var
done
```

#### 4. Create symbolic links for tensor pair from different samples
```bash
for s in "${!SOURCE_SAMPLE_NAMES[@]}"
do
  SAMPLE_NAME=${SOURCE_SAMPLE_NAMES[s]}
  echo "[INFO] Create Tensor Pair for sample ${SAMPLE_NAME}"

  SAMPLE_DIR="${TENSOR_PAIR_FOLDER_PATH}/${SAMPLE_NAME}"
  mkdir ${SAMPLE_DIR}

  SAMPLE_SOURCE_CHR=(${!ALL_SAMPLES_SOURCE_CHR[s]})
  SAMPLE_SOURCE_DEPTH=(${!ALL_SAMPLES_SOURCE_DEPTHS[s]})
  for j in "${!SAMPLE_SOURCE_DEPTH[@]}"
  do
    DEPTH=${SAMPLE_SOURCE_DEPTH[j]}
    mkdir ${SAMPLE_DIR}/${DEPTH}

    for i in "${!SAMPLE_SOURCE_CHR[@]}"
    do
      ln -s \
      ${SOURCE_DATASET_PATHS[s]}/tensor_pair/${DEPTH}/tensor_pair_${SAMPLE_SOURCE_CHR[i]} \
      ${SAMPLE_DIR}/${DEPTH}/tensor_pair_${SAMPLE_SOURCE_CHR[i]}
    done
  done
done
```

#### 5. Create variant information with prefix
```bash
for s in "${!SOURCE_SAMPLE_NAMES[@]}"
do
  SAMPLE_NAME=${SOURCE_SAMPLE_NAMES[s]}
  PREFIX=${SOURCE_SAMPLE_PREFIX[s]}

  SAMPLE_DIR="${VARIANT_FOLDER_PATH}/${SAMPLE_NAME}"
  cd ${SAMPLE_DIR}

  gzip -dc all_var | \
  awk -v pre="${PREFIX}" '{print pre $0}' | \
  pigz -c > all_var_prefixed
done

# concat all all_var_prefixed into one file
cd ${TARGET_DATASET_FOLDER_PATH}
cat var/*/all_var_prefixed > var/all_var_prefixed
```

#### 6. Create tensor_pair with prefix
```bash
for s in "${!SOURCE_SAMPLE_NAMES[@]}"
do
  SAMPLE_NAME=${SOURCE_SAMPLE_NAMES[s]}
  PREFIX=${SOURCE_SAMPLE_PREFIX[s]}
  SAMPLE_DIR="${TENSOR_PAIR_FOLDER_PATH}/${SAMPLE_NAME}"

  SAMPLE_SOURCE_CHR=(${!ALL_SAMPLES_SOURCE_CHR[s]})
  SAMPLE_SOURCE_DEPTH=(${!ALL_SAMPLES_SOURCE_DEPTHS[s]})

  for j in "${!SAMPLE_SOURCE_DEPTH[@]}"
  do
    DEPTH=${SAMPLE_SOURCE_DEPTH[j]}

    echo "[INFO] Add prefix for ${SAMPLE_NAME} on depth ${DEPTH}"
    for i in "${!SAMPLE_SOURCE_CHR[@]}"
    do
      TENSOR_PAIR_FILE_NAME="${SAMPLE_DIR}/${DEPTH}/tensor_pair_${SAMPLE_SOURCE_CHR[i]}"

      gzip -dc ${TENSOR_PAIR_FILE_NAME} | \
      awk -v pre="${PREFIX}" '{print pre $0}' | \
      pigz -c > ${TENSOR_PAIR_FILE_NAME}_prefixed
    done
  done
done
```

#### 7. Shuffle, split and compress shuffled tensors
```bash
cd ${TARGET_DATASET_FOLDER_PATH}

# first round shuffle
ls tensor_pair/*/*/tensor_pair*prefixed | \
parallel --joblog ./uncompress_tensors_round_1.log -j${THREADS} \
--line-buffer --shuf --verbose --compress gzip -dc ::: | \
parallel --joblog ./round_robin_cat_round_1.log -j${THREADS} \
--line-buffer --pipe -N1000 --no-keep-order --round-robin --compress cat | \
split -l ${ESTIMATED_SPLIT_NO_OF_LINES} \
--filter='shuf | pigz > $FILE.gz' -d - ${SHUFFLED_TENSORS_FILE_PATH}/round1_

# second round shuffle
ls ${SHUFFLED_TENSORS_FILE_PATH}/round1_* | \
parallel --joblog ./uncompress_tensors.log -j${THREADS_LOW} \
--line-buffer --shuf --verbose --compress gzip -dc ::: | \
parallel --joblog ./round_robin_cat.log -j${THREADS} \
--line-buffer --pipe -N1000 --no-keep-order --round-robin --compress cat | \
split -l ${ESTIMATED_SPLIT_NO_OF_LINES} \
--filter='shuf | pigz > $FILE.gz' -d - ${SHUFFLED_TENSORS_FILE_PATH}/split_
```
> - If shuffle is not very important to you, you may consider not shuffle or shuffle one round only
> - If have enough memory resources, welcome to increase # of threads for faster shuffling process.

#### 8. Create small bins for each shuffled tensors using `Tensor2Bin` submodule
```bash
ls ${SHUFFLED_TENSORS_FOLDER_PATH}/split_* | \
parallel --joblog ./tensor2Bin.log -j${THREADS_LOW} "python ${CLAIR} tensor2Bin \
--tensor_fn {} \
--var_fn ${VARIANT_FOLDER_PATH}/all_var_prefixed \
--bin_fn ${BINS_FOLDER_PATH}/{/.}.bin \
--allow_duplicate_chr_pos"
```

#### 9. Combine small bins into a training dataset using `CombineBins` submodule
```bash
python ${CLAIR} CombineBins
```

---

## Train a new model

### Setup variables for training commands afterwards
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"
MODEL_NAME=[YOUR_MODEL_NAME]                                 # e.g. "001"
MODEL_FOLDER_PATH="[YOUR_MODEL_FOLDER_PATH]/${MODEL_NAME}"   # will create this folder later
TENSOR_FILE_PATH=[YOUR_BIN_FILE_PATH]                        # e.g. ./tensor.bin

mkdir ${MODEL_FOLDER_PATH}

# set which gpu to run
export CUDA_VISIBLE_DEVICES="0"
```

### Train using `train` submodule
```bash
python $CLAIR train \
--bin_fn "$TENSOR_FILE_PATH" \
--ochk_prefix "${MODEL_FOLDER_PATH}/model"
```

### Train using `train_clr` submodule (using Cyclical Learning Rate (CLR))
```bash
# CLR modes: "tri"/"tri2"/"exp"
CLR_MODE=[CLR_MODE_OPTION]

python $CLAIR train_clr \
--bin_fn "$TENSOR_FILE_PATH" \
--ochk_prefix "${MODEL_FOLDER_PATH}/model" \
--clr_mode "$CLR_MODE" \
--SGDM
```
> For both `train` and `train_clr`:
> * Available Optimizers:
>   * `--Adam` (default)
>   * `--SGDM` (Stochastic Gradient Descent with momentum)
> * Available loss function:
>    * `--focal_loss` (default)
>    * `--cross_entropy`

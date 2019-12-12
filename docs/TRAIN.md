# Train a model for Clair

This document shows how to train a deep learning model for Clair.

## Prerequisitions
- A powerful GPU
    - RTX Titan (tested)
    - GTX 2080 Ti (tested)
    - GTX 1080 Ti (tested)
    - Any Nvidia card with 11GB memory or above will suffice, but speeds differ
- Clair installed
- GNU Parallel installed

## Catalogue
- [I. Preprocessing: Downsampling a sample](#i-preprocessing-downsampling-a-sample)
- [II. Build a compressed binary for training](#ii-build-a-compressed-binary-for-training)
    - [Single individual](#single-individual)
    - [Multiple individuals](#multiple-individuals)
- [III. Model training](#iii-model-training)

---

## I. Preprocessing: Downsampling a sample

To build a training dataset with multiple coverages, we need to create multiple downsampled BAM files from the original BAM file.

```bash
# please make sure the provided bam file is sorted and samtools indexed (e.g. hg001.bam)
BAM_FILE_PATH="[YOUR_BAM_FILE_PATH]"

# make sure the folder exists
SUBSAMPLED_BAMS_FOLDER_PATH="[SUBSAMPLED_BAMS_FOLDER_PATH]"

# FRAC values for 'samtools view -s INT.FRAC'
# please refer to samtools' documentation for further information
# in the exampled we set 80%, 40%, 20% and 10% of the full coverage
DEPTHS=(800 400 200 100)

# set to the number of CPU cores you have
THREADS=24

# downsampling
for i in "${!DEPTHS[@]}"
do
  samtools view -@ ${THREADS} -s ${i}.${DEPTHS[i]} -b ${BAM_FILE_PATH} \
  > ${SUBSAMPLED_BAMS_FOLDER_PATH}/0.${DEPTHS[i]}.bam
  samtools index -@ ${THREADS} ${SUBSAMPLED_BAMS_FOLDER_PATH}/0.${DEPTHS[i]}.bam
done

# add symbolic links for the orginal (full coverage) BAM
ln -s ${BAM_FILE_PATH} ${SUBSAMPLED_BAMS_FOLDER_PATH}/1.000.bam
ln -s ${BAM_FILE_PATH}.bai ${SUBSAMPLED_BAMS_FOLDER_PATH}/1.000.bam.bai
```

## II. Build a compressed binary for training

### Caveats
> - The whole procedure was break into blocks for better readability and error-tracing.
> - For each `parallel` command ran with the `--joblog` option, we can check the `Exitval` column from the job log output. If the column contains a non-zero value, it means error occured, please try to rerun the block again.
> - We suggest to use absolute path everywhere.

### Single indivdiual

This section shows how to build a compressed binary for one individual with or without multiple coverages.

#### 1. Setup variables for building bin
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"                               # e.g. clair.py
PYPY="[PYPY_BIN_PATH]"                                         # e.g. pypy3

VCF_FILE_PATH="[YOUR_VCF_FILE_PATH]"                           # e.g. hg001.vcf.gz
BAM_FILE_PATH="[YOUR_BAM_FILE_PATH]"                           # e.g. hg001.bam
REFERENCE_FILE_PATH="[YOUR_FASTA_FILE_PATH]"                   # e.g. hg001.fasta

# dataset output folder (the directory will be created later)
DATASET_FOLDER_PATH="[OUTPUT_DATASET_FOLDER_PATH]"

# array of coverages, (1.000) if downsampling was not used
DEPTHS=(1.000 0.800)

# where to find the BAMs prefixed as the elements in the DEPTHS array (e.g. 1.000.bam 0.800.bam)
# please refer to the `Preprocessing: Downsampling a sample` section
SUBSAMPLED_BAMS_FOLDER_PATH="[SUBSAMPLED_BAMS_FOLDER_PATH]"

# chromosome prefix ("chr" if chromosome names have the "chr"-prefix)
CHR_PREFIX=""

# array of chromosomes (do not include "chr"-prefix)
CHR=(21 22 X)

# number of cores to be used
THREADS=24

# for multiple memory intensive steps, this number of cores will be used
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

# create directories for different coverages
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

#### 3. Get truth variants using the `GetTruth` submodule
```bash
parallel --joblog ./get_truth.log -j${THREADS} \
"${PYPY} ${CLAIR} GetTruth \
--vcf_fn ${VCF_FILE_PATH} \
--var_fn ${VARIANT_FOLDER_PATH}/var_{1} \
--ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]}

# merge all truth variants into a single file (named all_var)
cat ${VARIANT_FOLDER_PATH}/var_* > ${VARIANT_FOLDER_PATH}/all_var
```

#### 4. Get random non-variant candidates using the `ExtractVariantCandidates` submodule
```bash
parallel --joblog ./evc.log -j${THREADS} \
"${PYPY} ${CLAIR} ExtractVariantCandidates \
--bam_fn ${BAM_FILE_PATH} \
--ref_fn ${REFERENCE_FILE_PATH} \
--can_fn ${CANDIDATE_FOLDER_PATH}/can_{1} \
--ctgName ${CHR_PREFIX}{1} \
--gen4Training" ::: ${CHR[@]}
```

#### 5. Create tensors for truth variants using the `CreateTensor` submodule
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

#### 6. Create tensors for non-variants using the `CreateTensor` submodule
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

#### 7. Merge truth variants and non-variants using the `PairWithNonVariants` submodule
```bash
parallel --joblog ./create_tensor_pair.log -j${THREADS} \
"${PYPY} ${CLAIR} PairWithNonVariants \
--tensor_can_fn ${TENSOR_CANDIDATE_FOLDER_PATH}/{1}/tensor_can_{2} \
--tensor_var_fn ${TENSOR_VARIANT_FOLDER_PATH}/{1}/tensor_var_{2} \
--output_fn ${TENSOR_PAIR_FOLDER_PATH}/{1}/tensor_pair_{2} \
--amp 2" ::: ${DEPTHS[@]} ::: ${CHR[@]}
```

#### 8. Shuffle, split and compress the tensors

##### One round shuffling (recommended)
```bash
ls tensor_pair/*/tensor_pair* | \
parallel --joblog ./uncompress_tensors.log -j${THREADS_LOW} -N2 \
--line-buffer --shuf --verbose --compress stdbuf -i0 -o0 -e0 pigz -p4 -dc ::: | \
parallel --joblog ./round_robin_cat.log -j${THREADS} \
--line-buffer --pipe -N1000 --no-keep-order --round-robin --compress \
"split - -l ${ESTIMATED_SPLIT_NO_OF_LINES} --filter='shuf | pigz -p4 > \$FILE.gz' -d ${SHUFFLED_TENSORS_FOLDER_PATH}/split_{#}_"
```

##### Two rounds shuffling (paranoid, but used in paper)
```bash
# the first round
ls tensor_pair/*/tensor_pair* | \
parallel --joblog ./uncompress_tensors_round_1.log -j${THREADS_LOW} -N2 \
--line-buffer --shuf --verbose --compress stdbuf -i0 -o0 -e0 pigz -p4 -dc ::: | \
parallel --joblog ./round_robin_cat_round_1.log -j${THREADS} \
--line-buffer --pipe -N1000 --no-keep-order --round-robin --compress \
"split - -l ${ESTIMATED_SPLIT_NO_OF_LINES} --filter='shuf | pigz -p4 > \$FILE.gz' -d ${SHUFFLED_TENSORS_FOLDER_PATH}/round1_{#}_"

# the second round
ls ${SHUFFLED_TENSORS_FOLDER_PATH}/round1_* | \
parallel --joblog ./uncompress_tensors_round_1.log -j${THREADS_LOW} -N2 \
--line-buffer --shuf --verbose --compress stdbuf -i0 -o0 -e0 pigz -p4 -dc ::: | \
parallel --joblog ./round_robin_cat.log -j${THREADS} \
--line-buffer --pipe -N1000 --no-keep-order --round-robin --compress \
"split - -l ${ESTIMATED_SPLIT_NO_OF_LINES} --filter='shuf | pigz -p4 > \$FILE.gz' -d ${SHUFFLED_TENSORS_FOLDER_PATH}/split_{#}_"
```

#### 9. Create splited binaries using the `Tensor2Bin` submodule
```bash
ls ${SHUFFLED_TENSORS_FOLDER_PATH}/split_* | \
parallel --joblog ./tensor2Bin.log -j${THREADS_LOW} \
"python ${CLAIR} Tensor2Bin \
--tensor_fn {} \
--var_fn ${VARIANT_FOLDER_PATH}/all_var \
--bin_fn ${BINS_FOLDER_PATH}/{/.}.bin \
--allow_duplicate_chr_pos"
```

#### 10. Merge splited binaries into a single binary using the `CombineBins` submodule
```bash
cd ${BINS_FOLDER_PATH}
python ${CLAIR} CombineBins
```

---

### Multiple individuals

This section shows how to build a binary of multiple individuals (genomes).

#### 0. For each individual, apply steps 1 to 7 in [Single individual](#single-individual).
__Please use different `DATASET_FOLDER_PATH` for different samples.__

#### 1. Setup variables
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"                               # e.g. clair.py
PYPY="[PYPY_BIN_PATH]"                                         # e.g. pypy3

# output folder
TARGET_DATASET_FOLDER_PATH="[TARGET_DATASET_FOLDER_PATH]"

# one line for each individual
SOURCE_SAMPLE_NAMES=(
  "hg001"
  "hg002"
)
# provide the DATASET_FOLDER_PATH directories created in `Single individual`.
SOURCE_DATASET_PATHS=(
  "[DATASET_FOLDER_PATH_FOR_SAMPLE_1]"
  "[DATASET_FOLDER_PATH_FOR_SAMPLE_2]"
)
# random prefix for each individuals
SOURCE_SAMPLE_PREFIX=(
  "u"
  "v"
)

# chromosome prefix ("chr" if chromosome names have the "chr"-prefix)
CHR_PREFIX=""

SAMPLE_1_DEPTHS=(1.000, 0.800)
SAMPLE_1_CHR=(21 22 X)

SAMPLE_2_DEPTHS=(1.000, 0.800)
SAMPLE_2_CHR=(21 22)

ALL_SAMPLES_SOURCE_DEPTHS=(
  SAMPLE_1_DEPTHS[@]
  SAMPLE_2_DEPTHS[@]
)

ALL_SAMPLES_SOURCE_CHR=(
  SAMPLE_1_CHR[@]
  SAMPLE_2_CHR[@]
)


NO_OF_SAMPLES=${#ALL_SAMPLES_SOURCE_DEPTHS[@]}
DEPTHS_PER_SAMPLE=${#SAMPLE_1_DEPTHS[@]}
ESTIMATED_SPLIT_NO_OF_LINES=$((90000 * $DEPTHS_PER_SAMPLE * $NO_OF_SAMPLES))

# number of cores to be used
THREADS=24

# for multiple memory intensive steps, this number of cores will be used
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

#### 3. Collect the truth variants from different individuals
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

#### 4. Collect the tensors from different individuals
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

#### 5. Add a random prefix to the truth variants of each individual
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

#### 6. Add a random prefix to the tensors of each individual
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

#### 7. Shuffle, split and compress the prefixed tensors
```bash
cd ${TARGET_DATASET_FOLDER_PATH}

ls tensor_pair/*/*/tensor_pair*prefixed | \
parallel --joblog ./uncompress_tensors.log -j${THREADS_LOW} -N2 \
--line-buffer --shuf --verbose --compress stdbuf -i0 -o0 -e0 pigz -p4 -dc ::: | \
parallel --joblog ./round_robin_cat.log -j${THREADS} \
--line-buffer --pipe -N1000 --no-keep-order --round-robin --compress \
"split - -l ${ESTIMATED_SPLIT_NO_OF_LINES} --filter='shuf | pigz -p4 > \$FILE.gz' -d ${SHUFFLED_TENSORS_FOLDER_PATH}/split_{#}_"
```

#### 8. Create splited binaries using the `Tensor2Bin` submodule
```bash
ls ${SHUFFLED_TENSORS_FOLDER_PATH}/split_* | \
parallel --joblog ./tensor2Bin.log -j${THREADS_LOW} "python ${CLAIR} Tensor2Bin \
--tensor_fn {} \
--var_fn ${VARIANT_FOLDER_PATH}/all_var_prefixed \
--bin_fn ${BINS_FOLDER_PATH}/{/.}.bin \
--allow_duplicate_chr_pos"
```

#### 9. Merge the splited binaries using the `CombineBins` submodule
```bash
cd ${BINS_FOLDER_PATH}
python ${CLAIR} CombineBins
```

---

## III. Model training

### Setup variables for the commands afterwards
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"
MODEL_NAME=[YOUR_MODEL_NAME]                                 # e.g. "001"
MODEL_FOLDER_PATH="[YOUR_MODEL_FOLDER_PATH]/${MODEL_NAME}" TENSOR_FILE_PATH=[YOUR_BIN_FILE_PATH]                        # e.g. ./tensor.bin

mkdir ${MODEL_FOLDER_PATH}

# set which gpu to use
export CUDA_VISIBLE_DEVICES="0"
```

### Start training using the `train` submodule (default: three times of learning rate decay)
```bash
python $CLAIR train \
--bin_fn "$TENSOR_FILE_PATH" \
--ochk_prefix "${MODEL_FOLDER_PATH}/model"
```

### Start training using `train_clr` submodule (use the Cyclical Learning Rate (CLR))
```bash
# CLR modes: "tri", "tri2" or "exp" (we suggest using "tri2")
CLR_MODE=[CLR_MODE_OPTION]

python $CLAIR train_clr \
--bin_fn "$TENSOR_FILE_PATH" \
--ochk_prefix "${MODEL_FOLDER_PATH}/model" \
--clr_mode "$CLR_MODE" \
--SGDM
```

### Other options
Applicable to both `train` and `train_clr`:

- Optimizer
 - `--Adam` (default)
 - `--SGDM` (Stochastic Gradient Descent with momentum)
- Loss function
 - `--focal_loss` (default)
 - `--cross_entropy`

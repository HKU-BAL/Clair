# Post Processing

This page shows scripts for the post-processing process.

## Prerequisites
- Clair installed
- GNU Parallel installed

## Catalogue
- [Handle overlapping variants](#handle-overlapping-variants)
- [Use multiple models for variant calling](#use-multiple-models-for-variant-calling)

---

## Handle overlapping variants

In the current implementation of Clair, each position of a given region is classified independently, so variants can overlap. For example, in vcf file output, two deletion variants overlap:
```bash
21  12345678  . TTTTATTTATTTA T
21  12345679  . TTTATTTATTTA T
```

At this moment there is a simple script to filter those variants by output variant with a higher QUAL score when two variants overlap.

```bash
zcat snp_and_indel.vcf.gz | \
python $CLAIR overlap_variant | \
bgziptabix snp_and_indel.filtered.vcf.gz
```

> Caveats: the true variant is possible to be removed by chance.

---

## Use multiple models for variant calling

If using Clair for variant calling on small # of positions, to leverage multiple models, here are the steps to combine the results of multiple models, or even many-models with many-bams.

#### 1. Setup
```bash
# each line represents an absolute path of a Clair model (one model is also fine)
CLAIR_MODELS=(
  "[YOUR_MODEL_FOLDER_PATH_1]"
  "[YOUR_MODEL_FOLDER_PATH_2]"
)
NO_OF_CLAIR_MODELS=${#CLAIR_MODELS[@]}

# make sure working directory exists
WORKING_DIRECTORY="[YOUR_WORKING_FOLDER_TO_STORE_ALL_OUTPUT]"

SAMPLE_NAME="[YOUR_SAMPLE_NAME]"

# set to the number of CPU cores you have
PARALLEL_THREADS="24"

# can use your own ways (e.g. downsample) to generate bunch of bams
# each line represents an absolute path of a bam (one bam is also fine)
BAM_FILE_PATHS=(
  "[YOUR_BAM_FILE_PATH_1]"
  "[YOUR_BAM_FILE_PATH_2]"
)

REFERENCE_FASTA_FILE_PATH="[YOUR_REFERENCE_FASTA_FILE]"  # e.g. chr21.fa
BED_FILE_PATH="[YOUR_BED_FILE]"                          # e.g. chr21.bed
```

#### 2. Output probabilities for each model-bam combination
```bash
INTERMEDIATE_OUTPUT_FOLDER="$WORKING_DIRECTORY/tmp_output"
mkdir $INTERMEDIATE_OUTPUT_FOLDER
cd $INTERMEDIATE_OUTPUT_FOLDER

for i in "${!BAM_FILE_PATHS[@]}"
do
  INPUT_BAM_FILE_PATH="${BAM_FILE_PATHS[i]}"

  BAM_PREFIX=`printf "%02d" $i`

  for j in "${!CLAIR_MODELS[@]}"
  do
    CLAIR_MODEL="${CLAIR_MODELS[j]}"
    MODEL_PREFIX=`printf "%02d" $j`
    SCRIPT_OUTPUT_FOLDER="m${MODEL_PREFIX}_b${BAM_PREFIX}"

    mkdir $SCRIPT_OUTPUT_FOLDER
    OUTPUT_PREFIX="$SCRIPT_OUTPUT_FOLDER/tmp"

    python $CLAIR callVarBamParallel \
    --chkpnt_fn "$CLAIR_MODEL" \
    --ref_fn "$REFERENCE_FASTA_FILE_PATH" \
    --bed_fn "$BED_FILE_PATH" \
    --bam_fn "$INPUT_BAM_FILE_PATH" \
    --pysam_for_all_indel_bases \
    --output_for_ensemble \
    --sampleName "$SAMPLE_NAME" \
    --output_prefix "$OUTPUT_PREFIX" > $SCRIPT_OUTPUT_FOLDER/call.sh
  done
done

cat */call.sh | parallel -j$PARALLEL_THREADS
```

#### 3. ensemble to create input for calling variants
```bash
FILES=(`ls m00_b00/*.vcf`)
ENSEMBLE_OUTPUT_FOLDER="$INTERMEDIATE_OUTPUT_FOLDER/ensemble"
mkdir $ENSEMBLE_OUTPUT_FOLDER
MININUM_NO_OF_VOTE_FOR_VARIANT="$(((${#BAM_FILE_PATHS[@]}*${#CLAIR_MODELS[@]}+2)/2))"
rm -f ensemble_command.sh

for i in "${!FILES[@]}"
do
  TARGET_FILE_NAME=`basename ${FILES[i]}`
  CAT_COMMAND=""
  for j in "${!BAM_FILE_PATHS[@]}"
  do
    BAM_PREFIX=`printf "%02d" $j`
    for k in "${!CLAIR_MODELS[@]}"
    do
      MODEL_PREFIX=`printf "%02d" $k`
      FOLDER_NAME="m${MODEL_PREFIX}_b${BAM_PREFIX}"
      CAT_COMMAND="$CAT_COMMAND $FOLDER_NAME/$TARGET_FILE_NAME"
    done
  done

  echo "cat ${CAT_COMMAND:1} | \
  python $CLAIR ensemble --minimum_count_to_output $MININUM_NO_OF_VOTE_FOR_VARIANT \
  > $ENSEMBLE_OUTPUT_FOLDER/$TARGET_FILE_NAME" >> ensemble_command.sh
done
cat ensemble_command.sh | parallel -j$PARALLEL_THREADS
```

#### 4. call_var with input from ensemble (output 1 vcf file per chr+region)
```bash
VCF_OUTPUT_FOLDER="$WORKING_DIRECTORY/output"
mkdir $VCF_OUTPUT_FOLDER
cd $WORKING_DIRECTORY
INPUT_FILES=(`ls tmp_output/ensemble/*.vcf`)
rm -f output.sh

for i in "${!INPUT_FILES[@]}"
do
  FILE_NAME=`basename ${INPUT_FILES[i]}`
  echo "cat tmp_output/ensemble/$FILE_NAME | \
  python $CLAIR call_var \
  --chkpnt_fn \"$CLAIR_MODEL\" \
  --ref_fn \"$REFERENCE_FASTA_FILE_PATH\" \
  --bam_fn \"$BAM_FILE_PATH\" \
  --call_fn \"$VCF_OUTPUT_FOLDER/$FILE_NAME\" \
  --sampleName \"$SAMPLE_NAME\" \
  --pysam_for_all_indel_bases \
  --input_probabilities" >> output.sh
done
cat output.sh | parallel -j$PARALLEL_THREADS
```

#### 5. Merge vcf files into one vcf file
```bash
cd $WORKING_DIRECTORY

vcfcat $VCF_OUTPUT_FOLDER/*.vcf | \
bcftools sort -m 2G | \
bgziptabix snp_and_indel.vcf.gz
```

#### 6. [OPTIONAL] remove temp files
```bash
cd $WORKING_DIRECTORY
rm -r $INTERMEDIATE_OUTPUT_FOLDER $VCF_OUTPUT_FOLDER
```

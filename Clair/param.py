NUM_THREADS = 12
maxEpoch = 10
parameterOutputPlaceHolder = 6

# Tensor related parameters, please use the same values for creating tensor, model training and variant calling
# Please change this value in the dataPrepScripts at the same time
flankingBaseNum = 16
matrixRow = 8
# Please change this value in the dataPrepScripts at the same time
matrixNum = 4
bloscBlockSize = 500

# Model hyperparameters
trainBatchSize = 5000
predictBatchSize = 10000
initialLearningRate = 2e-4
maximumLearningRate= 3e-3
learningRateDecay =1.08
maxLearningRateSwitch = 6
trainingDatasetPercentage = 0.9
decayStep=1
stepsizeConstant=1
clrGamma=0.99994

# Clair specific
l2RegularizationLambda = 0.005
l2RegularizationLambdaDecay = 1
dropoutRateFC4 = 0.5
dropoutRateFC5 = 0.0
dropoutRate = 0.05

# random seed (None to make it random for every run)
# set to None because cuDNN may introduce additional sources of randomness
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
RANDOM_SEED = None
OPERATION_SEED = None


def get_model_parameters():
    return dict(
        flankingBaseNum=16,
        matrixNum=4,
        expandReferenceRegion=1000000,
    )


# Global functions
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import sys
        raise sys.exit('Boolean value expected.')

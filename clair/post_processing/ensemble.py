from sys import stdin, stderr, argv, exit
from collections import namedtuple, defaultdict
from argparse import ArgumentParser

EnsembleConfig = namedtuple('EnsembleConfig', [
    'minimum_count_to_output',
])


def dicts_from_stdin():
    counter = defaultdict(lambda: 0)

    sequence_dict = {}
    tensor_dict = {}
    probabilities_dict = {}

    for row in stdin.readlines():
        columns = row.split(sep="\t")

        chromosome, position, sequence = columns[0], columns[1], columns[2]

        key = (chromosome, position)

        counter[key] = counter[key] + 1

        if not key in sequence_dict:
            sequence_dict[key] = sequence

        if not key in tensor_dict:
            tensor = [int(str_value) for str_value in columns[3:3 + 33*8*4]]
            tensor_dict[key] = tensor

        if not key in probabilities_dict:
            probabilities = [float(no) for no in columns[3+ 33*8*4:]]
            probabilities_dict[key] = probabilities
        else:
            probabilities_from_input = [float(no) for no in columns[3 + 33*8*4:]]

            probabilities = list.copy(probabilities_dict[key])
            for index, probability in enumerate(probabilities):
                probabilities[index] = probability + probabilities_from_input[index]

            probabilities_dict[key] = probabilities

    return counter, sequence_dict, tensor_dict, probabilities_dict


def output_with(
    counter,
    sequence_dict,
    tensor_dict,
    probabilities_dict,
    ensemble_config,
):
    minimum_count_to_output = ensemble_config.minimum_count_to_output

    for key, count in counter.items():
        if count < minimum_count_to_output:
            continue

        chromosome, position = key
        sequence = sequence_dict[key]
        tensor = tensor_dict[key]
        probabilities = probabilities_dict[key]

        tensor_str = "\t".join([str(int_value) for int_value in tensor])
        probabilities_str = "\t".join(["{:.6f}".format(probability / count) for probability in probabilities])

        print("\t".join([
            chromosome,
            position,
            sequence,
            tensor_str,
            probabilities_str,
        ]))


def run_pipeline(ensemble_config):
    counter, sequence_dict, tensor_dict, probabilities_dict = dicts_from_stdin()

    output_with(
        counter,
        sequence_dict,
        tensor_dict,
        probabilities_dict,
        ensemble_config,
    )


def main():
    parser = ArgumentParser(description="Call variants using a trained model and tensors of candididate variants")

    parser.add_argument('--minimum_count_to_output', type=int, default=0,
                        help="minimum # of calls to output the probabilities")

    args = parser.parse_args()

    if len(argv[1:]) == 0:
        parser.print_help()
        exit(1)

    ensemble_config = EnsembleConfig(
        minimum_count_to_output=args.minimum_count_to_output
    )
    run_pipeline(ensemble_config=ensemble_config)


if __name__ == "__main__":
    main()

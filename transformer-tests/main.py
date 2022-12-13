import random
import configargparse
from intrusion import Intrusion
from rating import Rating


def load_topics_rating(input_dir):
    file_path = input_dir + 'topics.txt'
    reader = open(file_path)
    # create list of list
    list_topics = [line.replace("\n", "") for line in reader.readlines()]

    return list_topics

def load_topics_intrusion(input_dir):
    lines = load_topics_rating(input_dir)
    topics = []
    # create list of dictionaries
    for idx, line in enumerate(lines):
        topics.append(
            {
                "topic_id": idx,
                "terms": line.split()
            }
        )
    return topics


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        description="parse args",
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    # Data
    parser.add("-c", "--config", is_config_file=True, default=None)
    parser.add("--input_dir", required=False, default=None)
    parser.add("--output_dir", required=False, default=None)
    parser.add("--data", required=True, default=None, choices=["20ng", "wiki", "nyt"])

    # Task-specific hyperparams
    parser.add("--task", default="intrusion", choices=["intrusion", "rating"])

    # Model-specific hyperparams
    parser.add("--model", required=True, default=None, choices=["mallet", "dvae", "etm"])
    parser.add("--num_topics", default=50, type=int)

    args = parser.parse_args()
    topic_file_path = './results/' + str(args.model) + '-' + str(args.data) +'/'
    path_save = './transformer-tests/' + str(args.model) + '/' + str(args.data) + '/'
    random.seed(5)
    if args.task == "intrusion":
        topics = load_topics_intrusion(topic_file_path)
        task_model = Intrusion(topics, args.num_topics, path_save, topic_file_path)
        task_model.run_intrusion()

    elif args.task == "rating":
        topics = load_topics_rating(topic_file_path)
        task_model = Rating(topics, args.num_topics, path_save, topic_file_path)
        task_model.run_rating()

    else:
        print("The task is not specified! Select the task to be rating or intrusion.")

    print('Done!')


import datetime
import os

from flashrag import pipeline
from flashrag.config import Config
from flashrag.dataset import Dataset
from flashrag.evaluator import Evaluator
from flashrag.utils import get_dataset
import argparse

#

def evaluate(args):
    config_dict = {"disable_save": True}
    # preparation
    config = Config(args.config_file, config_dict)
    dataset = Dataset(config, args.generated_dataset_path)
    evaluator = Evaluator(config)
    evaluator.save_metric_flag = False
    evaluator.save_data_flag = False
    # TODO handle pred_process_fun
    eval_result = evaluator.evaluate(dataset)
    print(eval_result)
    current_time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    evaluator.save_metric_score(eval_result, f"metric_score_{current_time_str}.txt")
    evaluator.save_data(dataset, f"evaluated_{current_time_str}.json")
    # evaluated_dataset_path = os.path.join(config["save_dir"], f"evaluated_{current_time_str}.json")
    # dataset.save(evaluated_dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating experiment")
    parser.add_argument("--generated-dataset-path", type=str, default=None, help="path to generated dataset for evaluation only")
    parser.add_argument("--config-file", type=str, default="my_config.yaml")
    args = parser.parse_args()
    evaluate(args)

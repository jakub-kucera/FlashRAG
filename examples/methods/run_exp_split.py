import os

from flashrag import pipeline
from flashrag.config import Config
from flashrag.dataset import Dataset
from flashrag.utils import get_dataset
import argparse


def naive(args):
    save_note = "naive"
    config_dict = {"save_note": save_note, "gpu_id": args.gpu_id, "dataset_name": args.dataset_name, "split": args.split}
    # disables creating new directory for evaluation
    config_dict["disable_save"] = args.evaluate_only

    # preparation
    config = Config(args.config_file, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    generated_dataset_path = args.generated_dataset_path

    from flashrag.pipeline import SequentialPipeline
    pipeline = SequentialPipeline(config)

    if not args.evaluate_only:
        result = pipeline.answer(test_data)
        generated_dataset_path = os.path.join(config["save_dir"], "generated.json")
        result.save(generated_dataset_path)

    if not args.generate_only:
        dataset = Dataset(config, generated_dataset_path)
        result = pipeline.evaluate(dataset)
        evaluated_dataset_path = os.path.join(config["save_dir"], "evaluated.json")
        result.save(evaluated_dataset_path)


def zero_shot(args):
    save_note = "zero-shot"
    config_dict = {"save_note": save_note, "gpu_id": args.gpu_id, "dataset_name": args.dataset_name, "split": args.split}
    # disables creating new directory for evaluation
    config_dict["disable_save"] = args.evaluate_only

    # preparation
    config = Config(args.config_file, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    generated_dataset_path = args.generated_dataset_path

    from flashrag.pipeline import ZeroShotPipeline
    from flashrag.prompt import PromptTemplate

    templete = PromptTemplate(
        config=config,
        system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
        user_prompt="Question: {question}",
    )
    pipeline = ZeroShotPipeline(config, templete)

    if not args.evaluate_only:
        result = pipeline.answer(test_data)
        generated_dataset_path = os.path.join(config["save_dir"], "generated.json")
        result.save(generated_dataset_path)

    if not args.generate_only:
        dataset = Dataset(config, generated_dataset_path)
        result = pipeline.evaluate(dataset)
        evaluated_dataset_path = os.path.join(config["save_dir"], "evaluated.json")
        result.save(evaluated_dataset_path)

    # result.sa


def selfrag(args):
    """
    Reference:
        Akari Asai et al. " SELF-RAG: Learning to Retrieve, Generate and Critique through self-reflection"
        in ICLR 2024.
        Official repo: https://github.com/AkariAsai/self-rag
    """
    config_dict = {
        "generator_model": "selfrag-llama2-7B",
        "generator_model_path": "model/selfrag_llama2_7b",
        "framework": "vllm",
        "save_note": "self-rag",
        "gpu_id": args.gpu_id,
        "generation_params": {
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "skip_special_tokens": False,
        },
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    # disables creating new directory for evaluation
    config_dict["disable_save"] = args.evaluate_only

    config = Config(args.config_file, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    generated_dataset_path = args.generated_dataset_path

    from flashrag.pipeline import SelfRAGPipeline

    pipeline = SelfRAGPipeline(
        config,
        threshold=0.2,
        max_depth=2,
        beam_width=2,
        w_rel=1.0,
        w_sup=1.0,
        w_use=1.0,
        use_grounding=True,
        use_utility=True,
        use_seqscore=True,
        ignore_cont=True,
        mode="adaptive_retrieval",
    )

    if not args.evaluate_only:
        result = pipeline.answer(test_data, long_form=False)
        generated_dataset_path = os.path.join(config["save_dir"], "generated.json")
        result.save(generated_dataset_path)

    if not args.generate_only:
        dataset = Dataset(config, generated_dataset_path)
        result = pipeline.evaluate(dataset)
        evaluated_dataset_path = os.path.join(config["save_dir"], "evaluated.json")
        result.save(evaluated_dataset_path)


def adaptive(args):
    judger_name = "adaptive-rag"
    model_path = "illuminoplanet/adaptive-rag-classifier"

    config_dict = {
        "judger_name": judger_name,
        "judger_config": {"model_path": model_path},
        "save_note": "adaptive-rag",
        "gpu_id": args.gpu_id,
        "dataset_name": args.dataset_name,
        "split": args.split,
    }
    # disables creating new directory for evaluation
    config_dict["disable_save"] = args.evaluate_only

    # preparation
    config = Config(args.config_file, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    generated_dataset_path = args.generated_dataset_path

    from flashrag.pipeline import AdaptivePipeline

    pipeline = AdaptivePipeline(config)
    # result = pipeline.run(test_data)

    if not args.evaluate_only:
        result = pipeline.answer(test_data)
        generated_dataset_path = os.path.join(config["save_dir"], "generated.json")
        result.save(generated_dataset_path)

    if not args.generate_only:
        dataset = Dataset(config, generated_dataset_path)
        result = pipeline.evaluate(dataset)
        evaluated_dataset_path = os.path.join(config["save_dir"], "evaluated.json")
        result.save(evaluated_dataset_path)


if __name__ == "__main__":
    func_dict = {
        "naive": naive,
        "zero-shot": zero_shot,
        "selfrag": selfrag,
        "adaptive": adaptive,
    }

    # TODO resolve config/dataset loading for evaluate only

    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--method_name", type=str, choices=func_dict.keys())
    parser.add_argument("--split", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--gpu_id", type=str)

    parser.add_argument("--generate-only", action="store_true", default=False)
    parser.add_argument("--generated-dataset-path", type=str, default=None, help="path to generated dataset for evaluation only")
    parser.add_argument("--evaluate-only", action="store_true", default=False)
    parser.add_argument("--config-file", type=str, default="my_config.yaml")
    # TODO add constraints
    #   generated_dataset_path only if generate-only
    #   evaluate-only XOR generate-only
    # parser.add_argument("--predictions-results-path", type=str, default=None)


    args = parser.parse_args()

    config_dict = {}
    config_dict["split"] = args.split
    config_dict["dataset_name"] = args.dataset_name
    config_dict["gpu_id"] = args.gpu_id

    func = func_dict[args.method_name]
    func(args)

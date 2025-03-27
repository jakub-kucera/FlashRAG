import json
import re
from copy import deepcopy

import numpy as np
import warnings
from collections import Counter
from flashrag.evaluator.utils import normalize_answer
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator


class BaseMetric:
    """`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    """

    metric_name = "base"

    def __init__(self, config):
        self.config = config
        self.dataset_name = config["dataset_name"]

    def calculate_metric(self, data):
        """Get the total score of this metric and score for each sample.

        Args:
            data object: it contains basic information and generated information.

        Returns:
            (metric_score: dict, metric_score_list: list)
            metric_score: such as ``{'em': 0.53}``.
            metric_score_list: score for each sample.

        """
        return {}, []

    def get_dataset_answer(self, data):
        if any(choice == [] for choice in data.choices):
            golden_answers_list = data.golden_answers
        else:
            # multi-choice dataset
            all_choices_list = data.choices
            golden_choice_idx_list = data.golden_answers
            golden_answers_list = [
                [choices[idx] for idx in idx_list]
                for choices, idx_list in zip(all_choices_list, golden_choice_idx_list)
            ]

        return golden_answers_list


class F1_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)

    def token_level_scores(self, prediction: str, ground_truths: list):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if (
                normalized_ground_truth in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth
            ):
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        f1 = sum(metric_score_list) / len(metric_score_list)
        return {"f1": f1}, metric_score_list


class Recall_Score(F1_Score):
    """Token-level Recall score"""

    metric_name = "recall"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"recall": precision}, metric_score_list


class Precision_Score(F1_Score):
    """Token-level Precision score"""

    metric_name = "precision"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"precision": precision}, metric_score_list


class ExactMatch(BaseMetric):
    r"""Exact match measure whether the predicted answer is completely consistent
    with the standard answer.

    """

    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return {"em": em_score}, metric_score_list


class Sub_ExactMatch(BaseMetric):
    r"""Sub-Exact match measure whether the predicted answer contains the standard answer."""

    metric_name = "acc"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_sub_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.search(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer in normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        sub_em_score = sum(metric_score_list) / len(metric_score_list)

        return {"acc": sub_em_score}, metric_score_list


class Retrieval_Recall(BaseMetric):
    r"""The recall of the top-k retreived passages, we measure if any of the passage contain the answer string."""

    metric_name = "retrieval_recall"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["metric_setting"]["retrieval_recall_topk"]

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        retrieve_docs = data.retrieval_result
        recall_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc["contents"] for doc in doc_list[: self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = 1 if any(hit_list) else 0
            recall_score_list.append(score)
        recall_score = sum(recall_score_list) / len(recall_score_list)

        return {f"retrieval_recall_top{self.topk}": recall_score}, recall_score_list


class Retrieval_Precision(BaseMetric):
    r"""The precision of the top-k retreived passages, we measure if any of the passage contain the answer string."""

    metric_name = "retrieval_precision"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["metric_setting"]["retrieval_recall_topk"]

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        retrieve_docs = data.retrieval_result
        precision_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc["contents"] for doc in doc_list[: self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = sum(hit_list) / len(hit_list)
            precision_score_list.append(score)
        precision_score = sum(precision_score_list) / len(precision_score_list)

        return {f"retrieval_precision_top{self.topk}": precision_score}, precision_score_list


class Rouge_Score(BaseMetric):
    metric_name = "rouge_score"
    cached_scores = {}

    def __init__(self, config):
        super().__init__(config)
        from rouge import Rouge

        self.scorer = Rouge()

    def calculate_rouge(self, pred, golden_answers):
        if (pred, tuple(golden_answers)) in self.cached_scores:
            return self.cached_scores[(pred, tuple(golden_answers))]
        output = {}
        for answer in golden_answers:
            scores = self.scorer.get_scores(pred, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]["f"])
        for k, v in output.items():
            output[k] = max(v)

        self.cached_scores[(pred, tuple(golden_answers))] = output
        return output


class Rouge_1(Rouge_Score):
    metric_name = "rouge-1"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-1": score}, metric_score_list


class Rouge_2(Rouge_Score):
    metric_name = "rouge-2"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-2"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-2": score}, metric_score_list


class Rouge_L(Rouge_Score):
    metric_name = "rouge-l"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-l"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-l": score}, metric_score_list


class ZH_Rouge_Score(BaseMetric):
    metric_name = "zh_rouge_score"
    cached_scores = {}

    def __init__(self, config):
        super().__init__(config)
        from rouge_chinese import Rouge

        self.scorer = Rouge()

    def calculate_rouge(self, pred, golden_answers):
        import jieba
        if (pred, tuple(golden_answers)) in self.cached_scores:
            return self.cached_scores[(pred, tuple(golden_answers))]
        output = {}
        pred = ' '.join(jieba.cut(pred))
        for answer in golden_answers:
            answer = ' '.join(jieba.cut(answer))
            scores = self.scorer.get_scores(pred, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]["f"])
        for k, v in output.items():
            output[k] = max(v)

        self.cached_scores[(pred, tuple(golden_answers))] = output
        return output


class ZH_Rouge_1(ZH_Rouge_Score):
    metric_name = "zh_rouge-1"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"zh_rouge-1": score}, metric_score_list


class ZH_Rouge_2(ZH_Rouge_Score):
    metric_name = "zh_rouge-2"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-2"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"zh_rouge-2": score}, metric_score_list


class ZH_Rouge_L(ZH_Rouge_Score):
    metric_name = "zh_rouge-l"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-l"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"zh_rouge-l": score}, metric_score_list


class BLEU(BaseMetric):
    metric_name = "bleu"

    def __init__(self, config):
        super().__init__(config)
        from ._bleu import Tokenizer13a

        self.tokenizer = Tokenizer13a()
        self.max_order = config["metric_setting"].get("bleu_max_order", 4)
        self.smooth = config["metric_setting"].get("bleu_smooth", False)

    def calculate_metric(self, data):
        from ._bleu import compute_bleu

        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        pred_list = [self.tokenizer(pred) for pred in pred_list]
        golden_answers_list = [
            [self.tokenizer(ans) for ans in golden_answers] for golden_answers in golden_answers_list
        ]
        score = compute_bleu(
            reference_corpus=golden_answers_list,
            translation_corpus=pred_list,
            max_order=self.max_order,
            smooth=self.smooth,
        )
        (total_bleu, precisions, bp, ratio, translation_length, reference_length) = score

        score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            pred = [pred]
            golden_answers = [golden_answers]
            score = compute_bleu(
                reference_corpus=golden_answers,
                translation_corpus=pred,
                max_order=self.max_order,
                smooth=self.smooth,
            )
            (bleu, precisions, bp, ratio, translation_length, reference_length) = score
            score_list.append(bleu)

        return {"bleu": total_bleu}, score_list


class LLMJudge(BaseMetric):
    metric_name = "llm_judge"
    # TODO wait, this does not check golden answers againts generated ones!!!
    JUDGE_PROMPT = """
    You will be given a user_question and system_answer couple.
    Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
    Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

    Provide your feedback as follows:

    Feedback:::
    Total rating: (your rating, as a float between 0 and 10)

    Now here are the question and answer.

    Question: {question}
    Answer: {answer}

    Feedback:::
    Total rating: """

    def __init__(self, config):
        super().__init__(config)
        if "llm_judge_setting" in config["metric_setting"]:
            llm_setting = config["metric_setting"]["llm_judge_setting"]
        else:
            assert False, "No available LLM settings!"
        # TODO: integrate generator class
        llm_name = llm_setting["model_name"]
        if "model_path" not in llm_setting:
            model_path = config["model2path"].get(llm_name, None)
        else:
            model_path = llm_setting["model_path"]
        if model_path is None:
            assert False, "None model path "

        from transformers import pipeline

        self.llm_pipeline = pipeline("text2text-generation", model=model_path, device=0)

    # TODO check this, answer is `self`???
    def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
        try:
            if split_str in answer:
                rating = answer.split(split_str)[1]
            else:
                rating = answer
            digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
            return float(digit_groups[0])
        except Exception as e:
            print(e)
            return 0

    def calculate_metric(self, data):
        question_list = data.question
        pred_list = data.pred

        judge_input_prompt = [self.JUDGE_PROMPT.format(question=q, answer=a) for q, a in zip(question_list, pred_list)]
        judge_output = self.llm_pipeline(judge_input_prompt, max_new_tokens=100, batch_size=8)
        judge_output = [item["generated_text"] for item in judge_output]
        # TODO store judge_output to dataset

        metric_score_list = [self.extract_judge_score(o) for o in judge_output]
        # rescale score
        metric_score_list = [score / 10 + 1 for score in metric_score_list]

        score = sum(metric_score_list) / len(metric_score_list)

        return {"llm_judge_score": score}, metric_score_list


class LLMJudgeMatcher(BaseMetric):
    metric_name = "llm_judge_matcher"

    SYSTEM_PROMPT = """
        Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the ground truth answer is present in the generated response by the following instructions:  
        1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
        2. Consider the substance of the answers – look for equivalent information or correct answers. Do
        not focus on exact wording unless the exact wording is crucial to the meaning.
        3. Your final decision should be based on whether the meaning and the vital facts of the "Ground
        truth" are present in the "Prediction" answer. For positive decision the score is 1, otherwise score is 0. 
          
        Output a valid JSON blob with a short "explanation" field explaining your answer as short as possible and a "score" field with value 1 or 0.  
        Do not forget about double quotes for keys and values in the JSON blob.  
        Do not wrap the JSON blob in any other text or ``` code block.
    """

    USER_PROMPT = "Question: '{question}'\n Ground truth: '{ground_truth}'\n Prediction: '{answer}'\n"

    def __init__(self, config):
        super().__init__(config)
        assert "llm_judge_generator_override" in config["metric_setting"], "No available LLM settings for LLM Judge!"
        self.overridden_config = deepcopy(config)
        if llm_setting := config["metric_setting"].get("llm_judge_generator_override", {}):
            self.overridden_config.final_config.update(llm_setting)
        self.generator = get_generator(self.overridden_config)
        self.prompt_template = PromptTemplate(self.overridden_config, system_prompt=self.SYSTEM_PROMPT, user_prompt=self.USER_PROMPT)

    def extract_judge_score(self, answer: str) -> int | None:
        try:
            answer_json = json.loads(answer)
            return int(answer_json["score"])
        except Exception as e:
            print(e)
            return None

    def extract_golden_answer(self, choices, golden_answers) -> str | None:
        # TODO move logic to dataset?
        try:
            if isinstance(golden_answers, list):
                # print(f"Multiple ground truths, taking first one, {golden_answers}")
                if isinstance(golden_answers[0], str):
                    # TODO handle multiple ground truths?
                    return golden_answers[0]
                elif isinstance(golden_answers[0], int):
                    if len(choices) == 0:
                        print("No choices available, but golden answer is index?")
                    else:
                        return choices[golden_answers[0]]
            elif isinstance(golden_answers, str):
                print(f"golden answer is string, {golden_answers}")
                return golden_answers
            elif isinstance(golden_answers, int):
                print(f"golden answer is int, {golden_answers}")
                if len(choices) == 0:
                    print("No choices available, but golden answer is index?")
                else:
                    return choices[golden_answers]
            else:
                print(f"golden answer is unknown type, {golden_answers}")
        except Exception as e:
            print(e)
            return None
        return None

    def calculate_metric(self, data):
        pred_list = data.pred
        question_list = data.question
        ground_truths_choice = [self.extract_golden_answer(c, ga) for c, ga in zip(data.choices, data.golden_answers)]

        ground_truths_extract_success = [0 if g is None else 1 for g in ground_truths_choice]
        data.update_output("ground_truths_extract_success", ground_truths_extract_success)

        assert len(question_list) == len(ground_truths_choice) == len(pred_list), "Length of inputs do not match"

        judge_input_prompts = [self.prompt_template.get_string(question=q, ground_truth=g, answer=a) for q, g, a in zip(question_list, ground_truths_choice, pred_list)]
        data.update_output("judge_input_prompt", judge_input_prompts)

        judge_output_list = self.generator.generate(judge_input_prompts)
        data.update_output("judge_output_raw", judge_output_list)

        metric_score_list = [self.extract_judge_score(i.judge_output_raw) for i in data]

        success_rate = [0 if score is None else 1 for score in metric_score_list]
        data.update_output("judge_output_success_rate", success_rate)

        # convert null values to 0
        metric_score_list_normalised = [0 if score is None else score for score in metric_score_list]
        data.update_output("judge_output_score", metric_score_list_normalised)

        # only count results, which valid extraction. Different size, cannot be saved
        adjusted_metric_score_list = [score for score in metric_score_list if score is not None]

        categories_scores_list = {}
        if "questions_type" in data[0].metadata:
            for item in data:
                if item.metadata["questions_type"] not in categories_scores_list:
                    categories_scores_list[item.metadata["questions_type"]] = []
                categories_scores_list[item.metadata["questions_type"]].append(item.judge_output_score)

        categories_scores = {f"llm_judge_matcher_accuracy_category_{key}": sum(v) / len(v) for key, v in categories_scores_list.items()}

        return {
            "llm_judge_matcher_accuracy": sum(metric_score_list_normalised) / len(metric_score_list_normalised),
            "llm_judge_matcher_adjusted_accuracy": sum(adjusted_metric_score_list) / len(adjusted_metric_score_list),
            "llm_judge_matcher_success_rate": sum(success_rate) / len(success_rate),
            "ground_truths_extract_success_rate": sum(ground_truths_extract_success) / len(ground_truths_extract_success),
            **categories_scores
        }, metric_score_list


class CountToken(BaseMetric):
    metric_name = "input_tokens"

    def __init__(self, config):
        super().__init__(config)
        tokenizer_name = config["metric_setting"].get("tokenizer_name", None)
        is_hf_tokenizer = True
        from flashrag.utils.constants import OPENAI_MODEL_DICT

        if tokenizer_name is None or tokenizer_name in OPENAI_MODEL_DICT:
            # use gpt4 tokenizer
            import tiktoken

            if tokenizer_name is None:
                tokenizer_name = "gpt-4"
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            is_hf_tokenizer = False
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer

    def calculate_metric(self, data):
        input_prompts = data.prompt
        if self.is_hf_tokenizer:
            token_counts = [len(self.tokenizer.tokenize(text)) for text in input_prompts]
        else:
            token_counts = [len(self.tokenizer.encode(text)) for text in input_prompts]
        avg_tokens = sum(token_counts) / len(token_counts)

        return {"avg_input_tokens": avg_tokens}, token_counts

class GAOKAOMM_Accuracy(BaseMetric):
    metric_name = 'gaokao_acc'
    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        metric_dict = {}
        acc_list = []
        for item in data:
            golden_answers = item.golden_answers
            golden_answers = [i.lower() for i in golden_answers]
            golden_answer = "".join(golden_answers)
            pred = item.pred.lower()
            subject = item.subject

            question_type = item.question_type
            if question_type == 'single_choice':
                acc = 1.0 if pred == golden_answer else 0.0
            else:
                if pred == golden_answer:
                    acc = 1.0
                elif pred in golden_answer:
                    acc = 0.5
                else:
                    acc = 0.0
            acc_list.append(acc)
            if subject not in metric_dict:
                metric_dict[subject] = []
            metric_dict[subject].append(acc)
        for key, value in metric_dict.items():
            metric_dict[key] = np.mean(value)

        metric_dict['avg_score'] = np.mean(acc_list)
        return metric_dict, acc_list

class AvgRetrievalCalls(BaseMetric):
    metric_name = "avg_retrieval_calls"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        avg_retrieval_calls = None
        try:
            retrieval_calls = data.retrieval_count
            avg_retrieval_calls = sum(retrieval_calls) / len(retrieval_calls)
        except Exception as e:
            print(f"{self.metric_name}: {e}")

        avg_retrieved_docs_count = None
        try:
            retrieved_docs = data.retrieval_result
            retrieved_docs_count = [len(d) for d  in retrieved_docs]
            avg_retrieved_docs_count = sum(retrieved_docs_count) / len(retrieved_docs_count)
        except Exception as e:
            print(f"{self.metric_name}: {e}")

        return {
                "avg_retrieval_calls": avg_retrieval_calls,
                "avg_retrieved_docs": avg_retrieved_docs_count,
            }, []

# TODO hallucination

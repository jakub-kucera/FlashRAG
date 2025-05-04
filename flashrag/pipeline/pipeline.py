import abc

from langchain.agents import Tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from tqdm import tqdm

from flashrag.dataset import Dataset
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.evaluator import Evaluator
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger


class BasicPipeline(metaclass=abc.ABCMeta):
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = None
        self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset, do_eval=True, pred_process_fun=None) -> Dataset:
        """The overall inference process of a RAG framework."""
        dataset = self.answer(dataset)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    @abc.abstractmethod
    def answer(self, dataset) -> Dataset:
        """The overall inference process of a RAG framework without final evaluation."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None) -> Dataset:
        """The evaluation process after finishing overall generation"""
        # load evaluator if not loaded
        if self.evaluator is None and do_eval:
            self.evaluator = Evaluator(self.config)

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class ZeroShotPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> generator
        """

        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        self.use_fid = config["use_fid"]

    def answer(self, dataset):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=i.question, choices=i.choices) for i in dataset]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset.update_output("retrieval_count", [0] * len(dataset))
        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG - kept for compatibility, but ZeroShotPipeline should be used instead
        input_prompts = [self.prompt_template.get_string(question=i.question, choices=i.choices) for i in dataset]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset.update_output("retrieval_count", [0] * len(dataset))

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def answer(self, dataset):
        input_query = dataset.question
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("retrieval_count", [1] * len(retrieval_results))

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=i.question, choices=i.choices, retrieval_result=i.retrieval_result)
                    for i in dataset]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=i.question, choices=i.choices,
                                                    formatted_reference=i.refine_results)
                        for i in dataset]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=i.question, choices=i.choices,
                                                    retrieval_result=i.retrieval_result)
                    for i in dataset
                ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc['contents'] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        return dataset


    def answer_leave_one_out(self, dataset):
        retrieval_results = []
        for i in dataset:
            retrieval_results.append(self.retriever.search_leave_1_out(query=i.question, file=i.metadata['file']))
        # self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("retrieval_count", [1] * len(retrieval_results))

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=i.question,
                                                    choices=i.choices,
                                                    formatted_references=self.prompt_template.format_weaviate_reference(i.retrieval_result))
                    for i in dataset]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=i.question, choices=i.choices,
                                                    formatted_reference=self.prompt_template.format_weaviate_reference(i.retrieval_result))
                        for i in dataset]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=i.question, choices=i.choices,
                                                    formatted_reference=self.prompt_template.format_weaviate_reference(i.retrieval_result))
                    for i in dataset
                ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc['contents'] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        return dataset



class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)

        self.judger = get_judger(config)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator
        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        self.sequential_pipeline = SequentialPipeline(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )

    def answer(self, dataset):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        pos_dataset, neg_dataset = dataset_split[True], dataset_split[False]

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        return dataset


class AdaptivePipeline(BasicPipeline):
    def __init__(
        self,
        config,
        norag_template=None,
        single_hop_prompt_template=None,
        multi_hop_prompt_template=None,
        retriever = None,
        generator = None
    ):
        super().__init__(config)
        # load adaptive classifier as judger
        self.judger = get_judger(config)

        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)

        self.generator = generator
        self.retriever = retriever

        # Load three pipeline for three types of query: naive/single-hop/multi-hop
        from flashrag.pipeline import IRCOTPipeline

        if norag_template is None:
            norag_templete = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
        self.norag_pipeline = SequentialPipeline(
            config,
            prompt_template=norag_templete,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline(
            config,
            prompt_template=single_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

        self.multi_hop_pipeline = IRCOTPipeline(
            config, prompt_template=multi_hop_prompt_template, retriever=retriever, generator=generator, max_iter=5
        )

    def answer(self, dataset):
        # judge_result: choice result representing which pipeline to use(e.g. A, B, C)
        # TODO edit to not load judge and generator models to memory at the same time
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        for symbol, symbol_dataset in dataset_split.items():
            if symbol == "A":
                symbol_dataset = self.norag_pipeline.naive_run(symbol_dataset, do_eval=False)
            elif symbol == "B":
                symbol_dataset = self.single_hop_pipeline.run(symbol_dataset, do_eval=False)
            elif symbol == "C":
                symbol_dataset = self.multi_hop_pipeline.run(symbol_dataset, do_eval=False)
            else:
                assert False, "Unknown symbol!"

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        return dataset


    def answer_leave_one_out(self, dataset):
        # judge_result: choice result representing which pipeline to use(e.g. A, B, C)
        # TODO edit to not load judge and generator models to memory at the same time
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        # dataset_split = split_dataset(dataset, judge_result)
        # for symbol, symbol_dataset in dataset_split.items():
        output_data = []
        for symbol, data in zip(judge_result, dataset.data):
            dataset_single = Dataset(config=dataset.config, data=[data])
            if symbol == "A":
                output_single_dataset = self.norag_pipeline.naive_run(dataset_single, do_eval=False)
            elif symbol == "B":
                output_single_dataset = self.single_hop_pipeline.answer_leave_one_out(dataset_single)
            elif symbol == "C":
                self.retriever.hide_data(data.metadata["file"])
                output_single_dataset = self.multi_hop_pipeline.run(dataset_single, do_eval=False)
                self.retriever.unhide_data(data.metadata["file"])
            else:
                assert False, "Unknown symbol!"
            output_data.append(output_single_dataset.data[0])

        # merge datasets into original format
        # dataset = merge_dataset(dataset_split, judge_result)
        output_dataset = Dataset(config=dataset.config, data=output_data)

        return output_dataset


class CorrectiveRAGPipeline(BasicPipeline):

        # based on https://docs.llamaindex.ai/en/stable/examples/workflow/corrective_rag_pack/
        retrieval_rating_prompt_str = """As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

                    Evaluation Criteria:
                    - Consider whether the document contains keywords or topics related to the user's question.
                    - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

                    Decision:
                    - Assign a binary score to indicate the document's relevance.
                    - Use 'yes' if the document is relevant to the question, or 'no' if it is not.

                    Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""

        # based on https://docs.llamaindex.ai/en/stable/examples/workflow/corrective_rag_pack/ but edited
        transform_web_query_prompt_str = """Your task is to refine a query to ensure it is highly effective for retrieving relevant web search results. \n
                Analyze the given input to grasp the core semantic intent or meaning. \n
                Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
                Respond with the optimized query only. Do not include any additional information or context. Only provide a single query."""

        def __init__(self, config, prompt_template=None, retriever=None, generator=None):
            """
            inference stage:
                query -> pre-retrieval -> retriever -> retrieval evaluation filtering -> query rewriting ->
                -> web search -> retrieval + web search merge -> generator
            """

            super().__init__(config, prompt_template)
            if generator is None:
                self.generator = get_generator(config)
            else:
                self.generator = generator

            if retriever is None:
                self.retriever = get_retriever(config)
            else:
                self.retriever = retriever

            # self.judger = get_judger(config)
            self.judger = self.generator

            self.retrieval_rating_prompt = PromptTemplate(config, system_prompt=self.retrieval_rating_prompt_str,
                                                          user_prompt="Question: {query_str}, Retrieved Document: {context_str}")
            self.transform_web_query_prompt = PromptTemplate(config, system_prompt=self.transform_web_query_prompt_str,
                                                                user_prompt="Original Query: {query_str}")

            if config["refiner_name"] is not None:
                self.refiner = get_refiner(config, self.retriever, self.generator)
            else:
                self.refiner = None

            from flashrag.retriever import TavilySearchRetriever
            self.web_retriever = TavilySearchRetriever(config=self.config)

        def answer(self, dataset, leave_one_out=False, enable_web=True):
            input_query = dataset.question
            # retrieval_results = [[
            #     "nothing found",
            #     "nothing found",
            # ]] * len(dataset)

            #  retrieval
            # retrieval_results = self.retriever.batch_search(input_query)
            retrieval_results = []
            for i in dataset:
                if leave_one_out:
                    retrieval_result = self.retriever.search_leave_1_out(query=i.question, file=i.metadata['file'])
                else:
                    retrieval_result = self.retriever.search(i.question)
                retrieval_results.append(retrieval_result)
            dataset.update_output("retrieval_result", retrieval_results)

            # rate retrieval
            questions = []
            for item in tqdm(dataset, desc="Rating retrieval results: "):
                for doc in item.retrieval_result:
                    questions.append(self.retrieval_rating_prompt.get_string(
                        context_str=doc, query_str=item.question))
            relevance_ratings_flat = self.generator.generate(questions)
            # unflatten list by nesting
            relevance_ratings = []
            for item in dataset:
                relevance_ratings_doc = []
                for _ in item.retrieval_result:
                    relevance_ratings_doc.append(relevance_ratings_flat.pop(0))
                relevance_ratings.append(relevance_ratings_doc)
            dataset.update_output("docs_relevance_ratings", relevance_ratings)

            # assign score to retrieval
            relevance_rating = []
            for item in dataset:
                if all(["yes" in rating.lower() for rating in item.docs_relevance_ratings]):

                    relevance_rating.append("correct")
                elif all("no" in rating.lower() for rating in item.docs_relevance_ratings):
                    relevance_rating.append("incorrect")
                else:
                    relevance_rating.append("ambiguous")
            dataset.update_output("relevance_rating", relevance_rating)

            # rephrase queries
            if enable_web:
                queries_to_rephrase = {}
                for c, item in enumerate(dataset):
                    if item.relevance_rating in ["incorrect", "ambiguous"]:
                        queries_to_rephrase[c] = item.question
                rephrased_queries = self.generator.generate(
                    [self.transform_web_query_prompt.get_string(query_str=query) for query in queries_to_rephrase.values()])

                web_search_queries = []
                for c, item in enumerate(dataset):
                    web_search_queries.append(rephrased_queries.pop(0) if c in queries_to_rephrase else None)
                dataset.update_output("web_search_query", web_search_queries)
            else:
                dataset.update_output("web_search_query", [None] * len(dataset))

            # web search
            web_search_results = []
            for c, item in tqdm(enumerate(dataset), desc="Web search: "):
                if item.web_search_query:
                    if len(item.web_search_query) < 4:
                        print(f"Invalid Web search query: {item.web_search_query}")
                        web_search_results.append(None)
                    elif len(item.web_search_query) >= 400:
                        print(f"Invalid Web search query: {item.web_search_query}")
                        shortened_query = item.web_search_query[:399]
                        web_search_results.append(shortened_query)
                    else:
                        web_search_results.append(self.web_retriever.search(item.web_search_query))
                else:
                    web_search_results.append(None)
            dataset.update_output("web_search_results", web_search_results)

            # merge retrieval and web search results
            input_prompts = []
            for c, item in enumerate(dataset):
                docs = [f"{doc['title']}: {doc['contents']}" for doc, rating in zip(item.retrieval_result, item.docs_relevance_ratings) if "yes" in rating.lower()]
                if item.web_search_results:
                    docs.extend(item.web_search_results)
                input_prompts.append(self.prompt_template.get_string(question=item.question, formatted_reference="\n".join(docs)))
            dataset.update_output("prompt", input_prompts)

            # final generation
            pred_answer_list = self.generator.generate(input_prompts)
            dataset.update_output("pred", pred_answer_list)

            return dataset


class ReActAgentPipeline(BasicPipeline):
    """
    Example pipeline that leverages LangChain's ReAct agent approach
    to handle question answering with a local LLaMA model.
    """

    def __init__(self,
                 config,
                 prompt_template: PromptTemplate = None,
                 retriever=None):
        """
        :param config: Dictionary with pipeline configuration.
                       Expect keys like "device", "model_path", etc.
        :param prompt_template: Optional custom prompt template (from flashrag).
        :param retriever: Optional custom retriever for retrieving documents.
        """
        super().__init__(config, prompt_template)

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        self.ollama_model_name = config['model2ollama'][config['generator_model']]

        self.llm = ChatOllama(
            model=self.ollama_model_name,
            temperature=0,
            num_ctx=config['generator_max_input_len']
        )

        self.retriever_calls = 0
        self.retrieval_results = []

        def retrieve_tool(query: str) -> str:
            if self.retriever is None:
                return "No retriever provided. Cannot find any relevant documents."

            docs = self.retriever.search(query)
            self.retriever_calls += 1
            self.retrieval_results.extend(docs)
            return "\n\n".join(
                [f"{doc['title']}: {doc['contents']}"
                 for doc in docs])

        retrieval_tool = Tool(
            name="retrieval_search",
            func=retrieve_tool,
            description="Use this tool to search for relevant documents based on a query. "
            "You can rephrase the query to be semantically similar to the documents you want to search for."
        )
        self.tools = [retrieval_tool]

        self.agent = create_react_agent(
            self.llm,
            tools=self.tools,
            prompt = f"""
                You are an assistant for question-answering tasks.
                Use three sentences maximum to answer the question in order to keep it concise.
                If you don't know some information needed to respond to users question or you are not sure, use the `{retrieval_tool.name}` tool to search for it.
                You can even use the search tool multiple times to search for different pieces of information.
                If you know the answer to the questions, answer immediately.
                """
        )


    def answer(self, dataset: Dataset) -> Dataset:
        """
        Implementation of answer() that uses the ReAct agent to produce answers.
        """
        for item in tqdm(dataset, desc="Agent processing questions: "):
            self.retriever_calls = 0
            self.retrieval_results = []

            question = item.question

            if choices := item.choices:
                user_prompt = f"""Question: '{question}'
                Pick the answer from one of the following choices: '{choices}'."""
            else:
                user_prompt =  f"question: `{question}`"

            for c, s in enumerate(self.agent.stream(input={"messages": [("user", user_prompt)]}, stream_mode="values"), start=1):
                message = s["messages"][-1]
            final_answer = message.content

            item.output["pred"] = final_answer
            item.output["retrieval_count"] = self.retriever_calls
            item.output["retrieval_result"] = self.retrieval_results
            item.output["agent_steps"] = c

            return dataset

    def answer_leave_one_out(self, dataset: Dataset) -> Dataset:
        """
        Implementation of answer() that uses the ReAct agent to produce answers.
        """
        for item in tqdm(dataset, desc="Agent processing questions: "):
            self.retriever_calls = 0
            self.retrieval_results = []

            self.retriever.hide_data(item.metadata["file"])

            question = item.question

            if choices := item.choices:
                user_prompt = f"""Question: '{question}'
                Pick the answer from one of the following choices: '{choices}'."""
            else:
                user_prompt = f"question: `{question}`"

            # for c, s in enumerate(self.agent.stream(input={"messages": [("user", user_prompt)]}, stream_mode="values"), start=1):
            #     message = s["messages"][-1]
            for c, s in enumerate(
                    self.agent.stream(input={"messages": [("user", user_prompt)]}, stream_mode="messages"),
                    start=1):
                message = s["messages"][-1]
            final_answer = message.content

            item.output["pred"] = final_answer
            item.output["retrieval_count"] = self.retriever_calls
            item.output["retrieval_result"] = self.retrieval_results
            item.output["agent_steps"] = c

            self.retriever.unhide_data(item.metadata["file"])

        return dataset

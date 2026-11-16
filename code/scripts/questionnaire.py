"""
Code developed to administer questionnaires to LLMs.

Please do not take this code as pedagogical material...

⚠️ Notes ⚠️ 
    This file is a bit monstruous, but everything was compiled in a single file for downloading and use ease.
    It is an agglomerate of code exctracted from: https://github.com/d-noe/question_llm/.  (Except for the visualisation)
    Please refer to this repository for more details. 

This file includes classes and functions to:
    - instantiate 'questionnaires' (make prompts, score results, etc.)
    - run inference with LLMs
    - administer questionnaires to LLMs (bridging both earlier classes)
    - visualise the results (single dimension only)
"""
# For questionnaires
import json
import torch
import gc
import numpy as np
from thefuzz import fuzz
from copy import deepcopy
from string import digits
from abc import abstractmethod
from string import ascii_lowercase as alc

# For visualisation
import pandas as pd
import altair as alt
from typing import List, Optional, Union

# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   LLM INTERACTION  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================

class LLMInference:
    def __init__(
        cls,
        model_id,
    ):
        cls.model_id = model_id

    @abstractmethod
    def prompt_model(
        cls,
        input_str,
        kwargs,
    ):
        raise NotImplementedError

    def __call__(
        cls,
        input_str,
        **kwargs,
    ):
        return cls.prompt_model(input_str, **kwargs)

class HFModelInference(LLMInference):
    def __init__(
        self,
        model_id,
        load_args:dict={},
        device:str = None,
        allow_cuda:bool=True,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        super().__init__(
            model_id
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_args)
        if device is None:
            if allow_cuda and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.model.to(self.device)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def prompt_model(
        self,
        input_str,
        logits_output:bool=True,
        generation_args={},
        tokenizer_args={}
    ):
        tokenized_prompt = self.tokenizer.apply_chat_template(
            [{"role":"user", "content":input_str}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **tokenizer_args
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                tokenized_prompt["input_ids"],
                output_scores=True,
                return_dict_in_generate=True,
                **generation_args,
            )
        if logits_output:
            model_response = output["scores"][0][0]
        else:
            model_response = self.tokenizer.decode(output["sequences"][0][tokenized_prompt["input_ids"].shape[-1]:])
        
        return model_response

# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~   HANDLE QUESTIONNAIRES  ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================

SURVEY_TEMPLATE = """{question}
{choices}"""

class Questionnaire:
    def __init__(
        self,
        categories:list,
        questions:list,
        choices:list,
        scores:list,
        
        index_type:str="alphabetical",
        choice_delim:str=". ",
        prompt_template:str=SURVEY_TEMPLATE,
        meta:dict={},
        #seed:int=None,
        **kwargs
    ):
        self.categories = categories
        
        self.questions = questions
        self.choices = choices
        self.scores = scores

        self.index_type = index_type
        self.choice_delim = choice_delim
        self.prompt_template = prompt_template

        self.meta = meta

        self._set_choices_index(
            self.index_type,
            inplace=True
        ) # sets -> self.indices & self.index_type

        self.categories_bias = [0]*len(self.categories)
        if "categories_bias" in kwargs.keys():
            self.categories_bias = kwargs["categories_bias"]
            kwargs.pop("categories_bias")

        seed = None
        if "seed" in kwargs.keys():
            seed = kwargs["seed"]
            kwargs.pop("seed")
        #seed = kwargs["seed"] if 'seed' in kwargs.keys() else None
        if not seed is None:
            self._shuffle_choices(seed=seed)

        self.meta = kwargs

    @classmethod
    def from_json(
        self,
        json_path,
        data_key:str = "data",
        **kwargs,
    ):
        with open(json_path) as f:
            loaded_survey = json.load(f)
        f.close()
        return self.from_dict(
            questionnaire_dict=loaded_survey,
            data_key=data_key,
            **kwargs
        )

    @classmethod
    def from_dict(
        self,
        questionnaire_dict,
        data_key = "data", 
        **kwargs,
    ):
        """
        format:
            survey = {
                "categories" : [...] ,
                "data" : {
                    "q_id" : {
                        "question" : ... ,
                        "choices" : [...] ,
                        "scores" : [...] ,
                    }
                }
            }
        
        return self(
            categories = questionnaire_dict["categories"],
            questions = [
                v["question"] 
                for v in questionnaire_dict["data"].values()
            ],
            choices = [
                v["choices"] 
                for v in questionnaire_dict["data"].values()
            ],
            scores = [
                v["scores"] 
                for v in questionnaire_dict["data"].values()
            ],
            **kwargs
        )
        """
        global_info = {
            k: v for k, v in questionnaire_dict.items()
            if not k==data_key
        }
        question_keys = list(
            questionnaire_dict[data_key].values()
        )[0].keys()
        question_info = {}
        for k in question_keys:
            if k.endswith('s'):
                question_info[k] = [v[k] for v in questionnaire_dict[data_key].values()]
            else:
                question_info["{}s".format(k)] = [v[k] for v in questionnaire_dict[data_key].values()]
        return self(
            **{**global_info, **question_info, **kwargs}
        )

    # =========
    
    def __getitem__(
        self,
        index
    ):
        return (
            self.questions[index], 
            self.choices[index], 
            self.scores[index]
        )
    
    def __len__(self):
        return len(self.questions)

    # =========

    def _set_choices_index(
        self,
        index_type:str,
        inplace:bool=False,
    ):
        if index_type=='alphabetical_l':
            indices = alc
        elif index_type=='alphabetical_u' or index_type=='alphabetical':
            indices = alc.upper()
        elif index_type=='numerical':
            indices = digits[1:]
        else:
            return NotImplementedError

        if inplace:
            self.index_type = index_type
            self.indices = list(indices)
        else:
            return list(indices)

    def _shuffle_choices(
        self,
        seed:int = None,
        inplace:bool=True,
    ):
        if not seed is None:
            np.random.seed(seed)
        n_ks = [len(cs) for cs in self.choices]
        reorder_indices = [
            np.random.choice(
                np.arange(len(cs)), len(cs), replace=False
            ).astype(int) for cs in self.choices
        ]

        rdm_choices = [
            [
                question_choices[rdm_id]
                for rdm_id in rdm_indices
            ]
            for question_choices, rdm_indices in zip(
                self.choices,
                reorder_indices
            )
        ]
        rdm_scores = [
            [
                question_scores[rdm_id]
                for rdm_id in rdm_indices
            ]
            for question_scores, rdm_indices in zip(
                self.scores,
                reorder_indices
            )
        ]

        if inplace:
            self.choices = rdm_choices#np.array(self.choices)[reorder_indices]
            self.scores = rdm_scores#np.array(self.scores)[reorder_indices]
            return 
        else:
            #return self.__init__(
            return type(self)(
                categories = self.categories,
                questions = self.questions,
                choices = rdm_choices,
                scores = rdm_scores,
                index_type = self.index_type,
                choice_delim = self.choice_delim,
                seed = None
            )

    # =========
    def make_prompts(
        self,
        shuffle_choices:bool=False,
    ):
        """
        prompt_template: "...{question}...{choices}..."
        """
        choices_str = [
            "\n".join(q_cs) if self.index_type is None else
            "\n".join([
                "{}{}{}".format(
                    self.indices[i],
                    self.choice_delim,
                    c
                )
                for i, c in enumerate(q_cs)
            ]) 
            for q_cs in self.choices
        ]
        prompts = [
            self.prompt_template.format(
                question = q,
                choices = cs
            )
            for q, cs in zip(self.questions, choices_str)
        ]
        
        return prompts

    def get_choices_keys(
        self,
    ):
        return [
            self.indices[:len(cs)]
            for cs in self.choices
        ]

    def _scores_to_dict(
        self,
    ):
        scores_dicts = [
            {
                choice_key: {
                    c:choice_score[i]
                    for i, c in enumerate(self.categories)
                }
                for choice_score, choice_key in zip(question_scores, question_choices)  # choice scores
            }
            for question_scores, question_choices in zip(self.scores, self.get_choices_keys()) # question scores
        ]
        return scores_dicts

    def evaluate(
        self,
        answers_probs:list,
        normalize_res:bool=False,
    ):
        """
        - TODO handle bias term! 
        answers_probs : [
            {...}, {...}
        ]
        """
        scores_dicts = self._scores_to_dict()
        results = {
            k:self.categories_bias[i]
            for i, k in enumerate(self.categories)
        }

        for choice_probs, question_scores in zip(answers_probs, scores_dicts):
            for choice_key, choice_prob in choice_probs.items():
                # check > 0 ? 
                for k in self.categories:
                    results[k] += question_scores[
                        choice_key # select scores for the considered answser
                    ][
                        k # select score for the considered category 
                    ]*choice_prob # weight by prob

        if normalize_res:
            tot_ = np.sum(list(results.values()))
            results = {k:v/tot_ for k,v in results.items()}
        # ? 
        """
        results["unclear"] = np.sum(
            [
                np.all([p==0 for p in question_as]) 
                for question_as in answers_ids
            ]
        )
        """
        return results

    # ========================
    def get_categories_scores(
        self,
        categories:list=None,
    ):
        if categories is None:
            categories = self.categories
        elif type(categories)==str:
            assert np.all([c in self.categories for c in categories])
            categories = [categories]
            
        categories_ids = [
            np.argmax([q_cat==cat for q_cat in self.categories])
            for cat in categories
        ]
            
        categories_scores = {
            cat: [
                [
                    (s[cat_id], a)
                    for a, s in zip(answers, scores)
                ]
                for _, answers, scores in self
            ] for cat_id, cat in zip(categories_ids, categories)
        }
            
        return categories_scores

    def _get_scores_range(
        self
    ):
        flat_scores = [
            s 
            for q_scores in self.scores 
            for qa_scores in q_scores
            for s in qa_scores
        ]
        return np.min(flat_scores), np.max(flat_scores)

    def get_range(
        self,
        category:str,
    ):
        cat_scores_answers = self.get_categories_scores([category])[category]
        cat_scores_numerical = [
            [sa[0] for sa in sa_q]
            for sa_q in cat_scores_answers
        ]
        return (np.sum([np.min(s) for s in cat_scores_numerical]), np.sum([np.max(s) for s in cat_scores_numerical]))

    def get_optim_answers(
        self,
        category:str,
        neg_examples:bool=False,
        thrs:float = 0,
    ):
        assert category in self.categories
        min_s, max_s = self._get_scores_range()
        pos_thres = max_s-thrs
        neg_thres = min_s+thrs

        cat_scores_answers = self.get_categories_scores([category])[category]
        optim_answers = [
            [
                sa[1]
                for sa in scores_answers
                if sa[0] >= pos_thres
            ]
            for scores_answers in cat_scores_answers
        ]

        if neg_examples:
            neg_optim_answers = [
                [
                    sa[1]
                    for sa in scores_answers
                    if sa[0] <= neg_thres
                ]
                for scores_answers in cat_scores_answers
            ]
            return optim_answers, neg_optim_answers
            
        return optim_answers


class AdministerQuestionnaire:
    def __init__(
        cls,
        questionnaire,
    ):
        cls.questionnaire = questionnaire
        cls.answers = None

    @abstractmethod
    def _get_answer_probs(
        cls,
        prompts
    ):
        raise NotImplementedError()

    def _set_answers(
        cls,
        answers
    ):
        cls.answers = answers
        return 
    
    def run(
        cls,
        **kwargs
    ):
        # 1. make prompts
        prompts = cls.questionnaire.make_prompts()
        # 2. get answers probabilities
        answers_probabilities = cls._get_answer_probs(prompts, **kwargs)
        cls._set_answers(answers_probabilities)
        # 3. eval 
        results = cls.questionnaire.evaluate(
            answers_probabilities
        )
        return results

class AdministerCustom(AdministerQuestionnaire):
    def __init__(
        self,
        questionnaire,
        generation_method,
        output_parser,
        generation_args={},
        parser_args={},
        store_answers:bool=True
    ):
        """
        generation_method 
            - takes as inputs:
                - prompt (str) 
                - (+ `generation_args`)
        output_parser
            - takes as inputs:
                - generation_method() output
                - choice keys
                - (+ `parser_args`)
            -> output
                - dict with keys: choice keys and values: probabilities
                    (eg. {"A": .2, "B": .7, "C": .1})
        """
        super().__init__(
            questionnaire
        )
        self.generation_method = generation_method
        self.generation_args = generation_args
        self.output_parser = output_parser
        self.parser_args = parser_args
        self.store_answers = store_answers
        if self.store_answers:
            self.generated_responses = []

    def _get_answer_probs(
        self,
        prompts,
        **kwargs,
    ):
        generated_responses = [
            self.generation_method(p,**self.generation_args) 
            for p in prompts
        ]
        if self.store_answers:
            self.generated_responses = generated_responses
        cks = self.questionnaire.get_choices_keys()
        copts = self.questionnaire.choices
        parsed_responses = [
            self.output_parser(r,cks[i],choices_strings=copts[i],**self.parser_args)
            for i, r in enumerate(generated_responses)
        ]

        return parsed_responses

class AdministerHF(AdministerCustom):
    def __init__(
        self,
        questionnaire,
        model_id:str,
        local:bool=False,
        logits_based:bool=False,
        load_args:dict={},
        generation_args:dict={},
        parser_args:dict={},
        store_answers:bool=True
    ):
        super().__init__(
            questionnaire,
            generation_method=self._generation_method,
            output_parser=self._parse_answer,
            generation_args=generation_args,
            store_answers=store_answers
        )
        if logits_based:
            assert local, "Not possible to retrieve logits from API. Set `logits_based` to False or run model locally."

        self.logits_based = logits_based
        self.local = local
        if self.local:
            self.inference_module = HFModelInference(
                model_id=model_id,
                load_args=load_args,
            )
        else:
            self.inference_module = HFSLAPIInference(
                model_id=model_id,
            )

    def _generation_method(
        self,
        input_str,
        **kwargs,
    ):
        inference_args = {
            "generation_args": self.generation_args,
        }
        if self.logits_based:
            inference_args["logits_output"]=True
        else:
            inference_args["logits_output"]=False
        return self.inference_module(
            input_str=input_str,
            **inference_args,
        )

    def _parse_answer(
        self,
        model_output,
        choices_keys:list,
        hard_scores:bool=False,
        default_to_fuzzy:bool=True,
        choices_strings:list=None,
    ):
        if self.logits_based:
            choice_ids = get_tokens_ids(
                self.inference_module.tokenizer,
                choices_keys,
                prefixes = [], # TODO ?
                suffixes = [], # TODO ?
                check_decode = False,
            )
            probs = get_tokens_prob(
                model_output, choice_ids, normalize = True,
            )
            if hard_scores:
                for k in probs.keys():
                    probs[k] = int(probs[k]==max(probs.values()))
        else:
            numerical_response = first_char_parser(model_output, numerical=self.questionnaire.index_type=="numerical")
            probs = {
                k: int(str(numerical_response).lower()==k.lower())
                for k in choices_keys
            }
            if (np.sum(list(probs.values()))==0) and (default_to_fuzzy) and (not choices_strings is None):
                fuzz_scores = [
                    fuzz.partial_ratio(model_output, c)
                    for c in choices_strings
                ]
                highest_fuzz_match = np.argmax(fuzz_scores)
                probs = {
                    k: int(i==highest_fuzz_match)
                    for i, k in enumerate(choices_keys)
                }
            
        return probs



# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  SOME UTILS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================

def first_char_parser(
    input_str,
    numerical:bool=True
):
    if numerical:
        try:
            return int(input_str.strip()[0])
        except:
            return 0 # np.Nan?
    else:
        return input_str.strip()[0]

def get_tokens_ids(
    tokenizer,
    inputs:list,
    prefixes = [],
    suffixes = [],
    token_id = -1,
    check_decode:bool=False # TODO
):
    tokens_ids_dict = {
        input: list(set(
            [tokenizer.encode(input)[token_id]]+[ # keep first token
                tokenizer.encode(pre+input)[token_id] 
                for pre in prefixes
            ]+[
                tokenizer.encode(input+suf)[token_id]
                for suf in suffixes
            ]
        ))
        for input in inputs
    }

    if check_decode:
        for k, values in tokens_ids_dict.items():
            tokens_ids_dict[k] = [
                v for v in values
                if k in tokenizer.decode(v)
            ]
    
    return tokens_ids_dict

def get_tokens_prob(
    logits,
    token_ids:list,
    normalize:bool=True,
):
    soft_m = torch.nn.functional.softmax(logits).to('cpu')[0]
    probs = {
        k: np.sum([soft_m[id] for id in ids])
        for k, ids in token_ids.items()
    }
    
    if normalize:
        tot = np.sum(list(probs.values()))
        for k in probs.keys():
            probs[k] = probs[k]/tot
    
    return probs


# ================================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  VISUALIZATION   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================================

# ------------------------------------------------------------
# Gradient Generator (lighter, with alpha)
# ------------------------------------------------------------
def make_gradient(min_score, max_score, left_color, right_color, alpha=0.5, height=60, steps=200):
    xs = np.linspace(min_score, max_score, steps)
    df = pd.DataFrame({
        "x1": xs[:-1],
        "x2": xs[1:],
        "color_x": xs[:-1],
    })

    return (
        alt.Chart(df)
        .mark_rect(opacity=alpha)   # ← lighter gradient
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=[min_score, max_score]), title=None),
            x2="x2:Q",
            color=alt.Color(
                "color_x:Q",
                scale=alt.Scale(
                    domain=[min_score, max_score],
                    range=[left_color, right_color]
                ),
                legend=None
            ),
        )
        .properties(height=height)
    )


# ------------------------------------------------------------
# Confidence Interval Band
# ------------------------------------------------------------
def make_confidence_band(low, high, min_score, max_score, alpha_ci=0.4):
    df = pd.DataFrame({"low": [low], "high": [high]})

    return (
        alt.Chart(df)
        .mark_rect(opacity=alpha_ci, color="grey")
        .encode(
            x=alt.X("low:Q", scale=alt.Scale(domain=[min_score, max_score])),
            x2="high:Q",
            tooltip=[
                alt.Tooltip("low:Q", title="CI Low"),
                alt.Tooltip("high:Q", title="CI High"),
            ]
        )
    )


# ------------------------------------------------------------
# Cloud of Scores (scatter)
# ------------------------------------------------------------
def make_score_cloud(scores, min_score, max_score, color="black"):
    df = pd.DataFrame({"score": scores})

    return (
        alt.Chart(df)
        .mark_circle(size=70, opacity=0.7, color=color)
        .encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[min_score, max_score])),
            tooltip=[alt.Tooltip("score:Q", title="Score")]
        )
    )


# ------------------------------------------------------------
# Markers (image or rule)
# ------------------------------------------------------------
def make_markers(
    scores,
    min_score,
    max_score,
    marker_color="black",
    marker_images=None,
    image_size:int=80,
    image_y_offset=40,  # negative values lift the image above the gradient
    model_names:list=None,
):
    """
    Draws markers for scores. Supports optional image markers, always draws a vertical rule line,
    and allows controlling the relative vertical position of the images.

    Parameters
    ----------
    scores : list[float] or float
        Scores to mark.
    min_score, max_score : float
        Axis limits.
    marker_color : str
        Color for the vertical rule line.
    marker_images : list[str] or None
        List of image URLs/paths for each score.
    image_y_offset : int
        Vertical offset for the image markers relative to the gradient (negative = above, positive = below)
    """
    if isinstance(scores, (int, float)):
        scores = [scores]

    df = pd.DataFrame({"score": scores})
    if not model_names is None:
        df["model"] = model_names
    layers = []

    # Vertical line for each score (always visible)
    tooltip = [alt.Tooltip("score:Q", title="Score")]
    if not model_names is None:
        tooltip += [alt.Tooltip("model:N", title="Model")]

    line_layer = (
        alt.Chart(df)
        .mark_rule(size=3, color=marker_color)
        .encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[min_score, max_score])),
            tooltip=tooltip
        )
    )
    layers.append(line_layer)

    # Add images if provided
    if marker_images:
        for i, (score, img) in enumerate(zip(scores, marker_images)):
            img_df = pd.DataFrame({"score": [score], "url": [img]})
            if not model_names is None:
                img_df["model"] = [model_names[i]]

            image_layer = (
                alt.Chart(img_df)
                .mark_image(
                    width=image_size,
                    height=image_size,
                    clip=False,
                    align="center",
                )
                .encode(
                    x=alt.X("score:Q", scale=alt.Scale(domain=[min_score, max_score])),
                    y=alt.value(image_y_offset),  # fixed vertical offset in pixels
                    url="url:N",
                    tooltip=tooltip
                )
            )

            layers.append(image_layer)

    return alt.layer(*layers)



# ------------------------------------------------------------
# Improved Axis Labels (just under title, aligned with borders)
# ------------------------------------------------------------
def make_axis_labels(min_score, max_score, left_label, right_label, height=60, fontsize=16):
    df_left = pd.DataFrame({
        "x": [min_score],
        "label": [f"← {left_label}"]
    })

    df_right = pd.DataFrame({
        "x": [max_score],
        "label": [f"{right_label} →"]
    })

    dy = height + fontsize/2

    left_text = (
        alt.Chart(df_left)
        .mark_text(
            dy=-dy,
            fontSize=fontsize,
            fontWeight="bold",
            align="left",
            clip=False
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[min_score, max_score])),
            text="label:N",
        )
    )

    right_text = (
        alt.Chart(df_right)
        .mark_text(
            dy=-dy,
            fontSize=fontsize,
            fontWeight="bold",
            align="right",
            clip=False
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[min_score, max_score])),
            text="label:N",
        )
    )

    return left_text + right_text



# ------------------------------------------------------------
# Middle Dashed Line
# ------------------------------------------------------------
def make_midline(min_score, max_score):
    mid = (min_score + max_score) / 2
    df = pd.DataFrame({"mid": [mid]})

    return (
        alt.Chart(df)
        .mark_rule(strokeDash=[6, 4], color="black", opacity=0.4)
        .encode(
            x=alt.X("mid:Q", scale=alt.Scale(domain=[min_score, max_score]))
        )
    )


# ------------------------------------------------------------
# MASTER FUNCTION
# ------------------------------------------------------------
def visualize_questionnaire_axis(
    category_name: str,
    scores: Union[float, List[float]],
    min_score: float,
    max_score: float,
    left_color: str = "#e8f6ff",
    right_color: str = "#075985",
    marker_color: str = "black",
    marker_images: Optional[List[str]] = None,
    image_size: int = 70,
    confidence_interval: Optional[tuple] = None,
    left_label: str = "Low",
    right_label: str = "High",
    width: int = 600,
    height: int = 60,
    title_font_size: int = 22,
    label_font_size: int = 16,
    interactive: bool = True,
    image_y_offset : float = None,
):
    if isinstance(scores, (int, float)):
        scores = [scores]

    layers = []

    # CI band under gradient
    if confidence_interval:
        low, high = confidence_interval
        layers.append(make_confidence_band(low, high, min_score, max_score))

    # Gradient
    layers.append(make_gradient(min_score, max_score, left_color, right_color, height=height))

    # Midline
    layers.append(make_midline(min_score, max_score))

    # Cloud points
    if len(scores) > 1:
        layers.append(make_score_cloud(scores, min_score, max_score, marker_color))

    # Markers
    if image_y_offset is None:
        image_y_offset = height/2
    layers.append(make_markers(scores, min_score, max_score, marker_color, marker_images, image_size=image_size, image_y_offset=image_y_offset))

    # Labels ABOVE gradient, INSIDE same chart
    layers.append(make_axis_labels(min_score, max_score, left_label, right_label, height=height/2, fontsize=label_font_size))

    chart = (
        alt.layer(*layers)
        .properties(
            width=width,
            height=height, 
            title=alt.TitleParams(
                category_name,
                fontSize=title_font_size,
                fontWeight="bold",
                anchor="middle"
            )
        )
    )

    if interactive:
        chart = chart.interactive()

    return chart


def visualize_multi_model_questionnaire_axis(
    category_name: str,
    results: dict,                  # {model: score}
    model_info: dict,               # {model: {image: str, color: str}}
    min_score: float,
    max_score: float,
    left_color="lightgray",
    right_color="black",
    left_label="Low",
    right_label="High",
    mode="facet",                   # "facet" = stacked; "overlay" = one single plot
    width=1000,
    height=100,
    image_size=50,
    interactive=True,
):
    """
    Visualize the same questionnaire category for multiple models
    either stacked or overlaid on a single gradient axis.
    """

    model_names = list(results.keys())

    # ======================
    # MODE 1: FACET / STACK
    # ======================
    if mode == "facet":

        charts = []

        for model in model_names:
            score = results[model]
            img = model_info[model]["logo"]
            color = model_info[model]["color"]

            chart = visualize_questionnaire_axis(
                category_name=f"{model} — {category_name}",
                scores=[score[category_name]],
                min_score=min_score,
                max_score=max_score,
                left_color=left_color,
                right_color=right_color,
                left_label=left_label,
                right_label=right_label,
                marker_color=color,
                marker_images=[img],
                width=width,
                height=height,
                image_size=image_size,
                interactive=interactive,
            )

            charts.append(chart)

        # Vertically concatenated plots
        final_chart = alt.vconcat(*charts)
        return final_chart


    # ======================
    # MODE 2: OVERLAY
    # ======================
    elif mode == "overlay":

        layers = []

        # shared gradient background (draw only once)
        gradient = make_gradient(min_score, max_score, left_color, right_color, height=height)
        midline = make_midline(min_score, max_score)
        labels = make_axis_labels(min_score, max_score, left_label, right_label)

        layers.extend([gradient, midline, labels])

        # Overlaid markers for all models
        for model in model_names:
            score = results[model]
            img = model_info[model]["logo"]
            color = model_info[model]["color"]

            layer = make_markers(
                scores=[score[category_name]],
                min_score=min_score,
                max_score=max_score,
                marker_color=color,
                marker_images=[img],
                image_y_offset=-40,
                model_names = [model]
            )
            layers.append(layer)

        final_chart = (
            alt.layer(*layers)
            .properties(
                width=width,
                height=height,
                title=alt.TitleParams(
                    f"{category_name}",
                    anchor="middle",
                    fontSize=22,
                    fontWeight="bold",
                )
            )
        )
        
        if interactive:
            final_chart = final_chart.interactive()

        return final_chart

    else:
        raise ValueError("mode must be 'facet' or 'overlay'")


def make_2d_axis_labels(
    min_score_x, max_score_x, 
    left_label, right_label, 
    min_score_y, max_score_y, 
    bottom_label, top_label, 
    fontsize=16
):
    df_left = pd.DataFrame({
        "x": [min_score_x],
        "y": [np.mean([min_score_y, max_score_y])],
        "label": [f"← {left_label}"]
    })
    df_right = pd.DataFrame({
        "x": [max_score_x],
        "y": [np.mean([min_score_y, max_score_y])],
        "label": [f"{right_label} →"]
    })
    df_top = pd.DataFrame({
        "y": [max_score_y],
        "x": [np.mean([min_score_x, max_score_x])],
        "label": [f"← {top_label}"]
    })
    df_bottom = pd.DataFrame({
        "y": [min_score_y],
        "x": [np.mean([min_score_x, max_score_x])],
        "label": [f"{bottom_label} →"]
    })

    left_text = (
        alt.Chart(df_left)
        .mark_text(
            fontSize=fontsize,
            fontWeight="bold",
            align="left",
            clip=False
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[min_score_x, max_score_x])),
            y="y:Q",
            text="label:N",
        )
    )
    right_text = (
        alt.Chart(df_right)
        .mark_text(
            fontSize=fontsize,
            fontWeight="bold",
            align="right",
            clip=False
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[min_score_x, max_score_x])),
            y="y:Q",
            text="label:N",
        )
    )

    top_text = (
        alt.Chart(df_top)
        .mark_text(
            fontSize=fontsize,
            fontWeight="bold",
            align="left",
            clip=False,
            angle=90,
        )
        .encode(
            x="x:Q",
            y=alt.X("y:Q", scale=alt.Scale(domain=[min_score_y, max_score_y])),
            text="label:N",
        )
    )
    bottom_text = (
        alt.Chart(df_bottom)
        .mark_text(
            fontSize=fontsize,
            fontWeight="bold",
            align="right",
            clip=False,
            angle=90,
        )
        .encode(
            x="x:Q",
            y=alt.X("y:Q", scale=alt.Scale(domain=[min_score_y, max_score_y])),
            text="label:N",
        )
    )

    return left_text + right_text + top_text + bottom_text

def visualize_two_axis_map(
    category_x,
    category_y,
    results,                 # dict: {model: {category: score}}
    model_info,              # dict: {model: {image: str, color: str}}
    x_range,                 # (min_x, max_x)
    y_range,                 # (min_y, max_y)
    x_labels,
    y_labels,
    quadrant_colors=None,    # dict: {'Q1':..., 'Q2':..., 'Q3':..., 'Q4':...}
    width=500,
    height=500,
    image_size=60,
    show_images=True,
):
    """
    2D axis plot with quadrant colors and legend-based interactive model highlighting.
    """

    # -------------------------
    # Quadrant colors defaults
    # -------------------------
    if quadrant_colors is None:
        quadrant_colors = {
            "Q1": "#f4cccc",  # top-right
            "Q2": "#c9daf8",  # top-left
            "Q3": "#d0e0e3",  # bottom-left
            "Q4": "#fce5cd",  # bottom-right
        }

    min_x, max_x = x_range
    min_y, max_y = y_range
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    # -------------------------
    # Dataframe of models
    # -------------------------
    rows = []
    for model, scores in results.items():
        rows.append({
            "model": model,
            "x": scores[category_x],
            "y": scores[category_y],
            "image": model_info[model]["logo"],
            "color": model_info[model]["color"],
        })

    df = pd.DataFrame(rows)

    # -------------------------
    # Quadrant background rectangles
    # -------------------------
    quads = pd.DataFrame([
        {"x1": min_x, "x2": mid_x, "y1": mid_y, "y2": max_y, "quad": "Q2"},
        {"x1": mid_x, "x2": max_x, "y1": mid_y, "y2": max_y, "quad": "Q1"},
        {"x1": min_x, "x2": mid_x, "y1": min_y, "y2": mid_y, "quad": "Q3"},
        {"x1": mid_x, "x2": max_x, "y1": min_y, "y2": mid_y, "quad": "Q4"},
    ])

    quad_layer = (
        alt.Chart(quads)
        .mark_rect()
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=[min_x, max_x]), title=None),
            x2="x2:Q",
            y=alt.X("y1:Q", scale=alt.Scale(domain=[min_y, max_y]), title=None),
            y2="y2:Q",
            color=alt.Color(
                "quad:N",
                scale=alt.Scale(
                    domain=["Q1", "Q2", "Q3", "Q4"],
                    range=[
                        quadrant_colors["Q1"],
                        quadrant_colors["Q2"],
                        quadrant_colors["Q3"],
                        quadrant_colors["Q4"],
                    ],
                ),
                legend=None,
            ),
        )
    )

    # -------------------------
    # Midlines
    # -------------------------
    dummy = pd.DataFrame([{"x": mid_x, "y": mid_y}])

    midline_x = (
        alt.Chart(dummy)
        .mark_rule(color="black", strokeWidth=1)
        .encode(x="x:Q")
    )

    midline_y = (
        alt.Chart(dummy)
        .mark_rule(color="black", strokeWidth=1)
        .encode(y="y:Q")
    )

    # -------------------------
    # Interactive legend selection
    # -------------------------
    selection = alt.selection_point(
        fields=["model"],
        bind="legend"   # makes the legend clickable/toggleable
    )

    # -------------------------
    # Model markers
    # -------------------------
    if show_images:
        marker_layer = (
            alt.Chart(df)
            .mark_image(width=image_size, height=image_size)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[min_x, max_x])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[min_y, max_y])),
                url="image:N",
                opacity=alt.condition(selection, alt.value(1), alt.value(0.25)),
                color=alt.Color("model:N", legend=alt.Legend(title="Models")),
                tooltip=[
                    alt.Tooltip("model:N"),
                    alt.Tooltip("x:Q", title=category_x),
                    alt.Tooltip("y:Q", title=category_y),
                ],
            )
            .add_params(selection)
        )
    else:
        marker_layer = (
            alt.Chart(df)
            .mark_circle(size=300)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[min_x, max_x])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[min_y, max_y])),
                color=alt.Color("model:N", legend=alt.Legend(title="Models")),
                opacity=alt.condition(selection, alt.value(1.0), alt.value(0.25)),
                tooltip=[
                    alt.Tooltip("model:N"),
                    alt.Tooltip("x:Q", title=category_x),
                    alt.Tooltip("y:Q", title=category_y),
                ],
            )
            .add_params(selection)
        )

    labels_layer = make_2d_axis_labels(
        min_score_x=x_range[0], max_score_x=x_range[1], 
        left_label=x_labels[0], right_label=x_labels[1], 
        min_score_y=y_range[0], max_score_y=y_range[1], 
        bottom_label=y_labels[0], top_label=y_labels[1], 
    )

    # -------------------------
    # Compose final chart
    # -------------------------
    chart = (
        alt.layer(
            quad_layer,
            midline_x,
            midline_y,
            marker_layer,
            labels_layer
        )
        .properties(
            width=width,
            height=height,
            title=f"{category_x} vs {category_y}"
        )
    )

    return chart

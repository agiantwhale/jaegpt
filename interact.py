# # Copyright (c) 2019-present, HuggingFace Inc.
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import HfArgumentParser, set_seed
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pipelines import base


@dataclass
class Arguments:
    model_checkpoint: Optional[str] = field(default="microsoft/DialoGPT-small")
    baseline_checkpoint: Optional[str] = field(default="microsoft/DialoGPT-small")
    max_history: Optional[int] = field(default=4)
    seed: Optional[int] = field(default=0xDEADBEAF)
    max_length: Optional[int] = field(default=20)
    min_length: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=True)
    early_stopping: Optional[bool] = field(default=None)
    num_beams: Optional[int] = field(default=None)
    temperature: Optional[float] = field(default=0.8)
    top_k: Optional[int] = field(default=100)
    top_p: Optional[float] = field(default=0.25)
    repetition_penalty: Optional[float] = field(default=None)
    bad_words_ids: Optional[Iterable[int]] = field(default=None)
    length_penalty: Optional[float] = field(default=None)
    no_repeat_ngram_size: Optional[int] = field(default=3)
    encoder_no_repeat_ngram_size: Optional[int] = field(default=None)
    num_return_sequences: Optional[int] = field(default=None)
    max_time: Optional[float] = field(default=None)
    max_new_tokens: Optional[int] = field(default=None)
    decoder_start_token_id: Optional[int] = field(default=None)
    use_cache: Optional[bool] = field(default=None)
    num_beam_groups: Optional[int] = field(default=None)
    diversity_penalty: Optional[float] = field(default=None)
    output_attentions: Optional[bool] = field(default=None)
    output_hidden_states: Optional[bool] = field(default=None)
    output_scores: Optional[bool] = field(default=None)
    return_dict_in_generate: Optional[bool] = field(default=None)
    forced_bos_token_id: Optional[int] = field(default=None)
    forced_eos_token_id: Optional[int] = field(default=None)
    remove_invalid_values: Optional[bool] = field(default=None)
    synced_gpus: Optional[bool] = field(default=None)
    use_token_type_ids: Optional[bool] = field(default=True)


def gen_response(
    args: Arguments,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    chat_history: List[str],
    baseline: bool = False,
):
    input_chat_history = chat_history
    if len(chat_history) >= args.max_history:
        input_chat_history = chat_history[-args.max_history :]

    history_toks = [
        tokenizer.encode(hist) + [tokenizer.eos_token_id] for hist in input_chat_history
    ]

    token_type_ids = sum(
        [[0 if i % 2 == 0 else 1] * len(toks) for i, toks in enumerate(history_toks)],
        [],
    )
    token_type_ids[-1] = 1

    new_user_input_ids = torch.tensor(sum(history_toks, [])).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)

    assert token_type_ids.shape == new_user_input_ids.shape

    bot_input_ids = new_user_input_ids

    if baseline:
        gen_args = {}
        gen_args["max_length"] = bot_input_ids.shape[-1] + 1000
    else:
        gen_args = vars(args).copy()
        gen_args["max_length"] += bot_input_ids.shape[-1]
        if args.use_token_type_ids:
            gen_args["token_type_ids"] = token_type_ids
    gen_args["pad_token_id"] = tokenizer.eos_token_id

    response_ids = model.generate(bot_input_ids, **gen_args)
    response = tokenizer.decode(
        response_ids[:, bot_input_ids.shape[-1] :][0],
        skip_special_tokens=True,
    )

    return response


def save_results(
    filepath: str,
    args: Arguments,
    cand_chat_history: List[str],
    base_chat_history: List[str],
):
    result = {
        "parameters": vars(args),
        "candidate_log": cand_chat_history,
        "baseline_log": base_chat_history,
    }

    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.writelines(
            json.dumps(
                result,
                indent=4,
                sort_keys=True,
            )
        )


def run():
    parser = HfArgumentParser((Arguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    args: Arguments

    if args.model_checkpoint == "":
        raise ValueError(
            "Interacting with GPT2 requires passing a finetuned model_checkpoint"
        )

    set_seed(args.seed)

    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    print(f"Loading {args.model_checkpoint} as fine-tuned model.")
    print(f"Loading {args.baseline_checkpoint} as baseline model.")
    tokenizer_class, model_class = GPT2Tokenizer, GPT2LMHeadModel
    cand_tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    cand_model = model_class.from_pretrained(args.model_checkpoint)
    base_tokenizer = tokenizer_class.from_pretrained(args.baseline_checkpoint)
    base_model = model_class.from_pretrained(args.baseline_checkpoint)

    cand_chat_history = []
    base_chat_history = []

    while True:
        print()
        print()
        raw_text = input(">>> ")
        while not raw_text:
            print("Prompt should not be empty!")
            print()
            raw_text = input(">>> ")

        if raw_text.startswith("save") and len(raw_text.split()) == 2:
            _, save_path = raw_text.split()
            save_results(
                filepath=save_path,
                args=args,
                cand_chat_history=cand_chat_history,
                base_chat_history=base_chat_history,
            )

            print()
            print()
            print(f"{'Saved results to':>20} {save_path}")
        else:
            cand_chat_history.append(raw_text)
            base_chat_history.append(raw_text)

            cand_response = gen_response(
                args=args,
                model=cand_model,
                tokenizer=cand_tokenizer,
                chat_history=cand_chat_history,
            )
            base_response = gen_response(
                args=args,
                model=base_model,
                tokenizer=base_tokenizer,
                chat_history=base_chat_history,
                baseline=True,
            )

            cand_chat_history.append(cand_response)
            base_chat_history.append(base_response)

            print()
            print()
            print(f"{'Fine-tuned':>20}: {cand_response}")
            print(f"{'Baseline':>20}: {base_response}")


if __name__ == "__main__":
    run()

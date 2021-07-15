from collections import defaultdict
from transformers.tokenization_utils import PreTrainedTokenizerBase
from typing import List


def tokenize_function(
    examples, tokenizer: PreTrainedTokenizerBase, block_size: int, pad: bool = True
):
    eos_id = tokenizer.eos_token_id
    pad_id = eos_id

    distractor_batch, history_batch, reply_batch = (
        examples["distractor"],
        examples["history"],
        examples["reply"],
    )

    outputs_data = defaultdict(list)

    for history, reply, distractors in zip(
        history_batch, reply_batch, distractor_batch
    ):
        hist_toks: List[List[int]] = [
            tokenizer.encode(hist["content"], add_special_tokens=False) + [eos_id]
            for hist in history
        ]
        hist_tok: List[int] = sum(hist_toks, [])

        cand_toks: List[List[int]] = [
            tokenizer.encode(cand, add_special_tokens=False)
            for cand in distractors + [reply]
        ]

        gold_index = len(cand_toks) - 1
        dummy = any(len(hist_tok) + len(cand) > block_size for cand in cand_toks)

        dataset = defaultdict(list)

        if dummy:
            for _ in cand_toks:
                dataset["input_ids"].append([pad_id] * block_size)
                dataset["token_type_ids"].append([0] * block_size)
                dataset["mc_token_ids"].append(0)
                dataset["label_ids"].append([-100] * block_size)

            dataset["mc_labels"].append(-100)
            dataset["overflow"] = True
        else:
            for i_cand, cand_tok in enumerate(cand_toks):
                dialog_toks = hist_toks + [cand_tok]
                content_toks = sum(dialog_toks, [])

                if pad:
                    padding = [pad_id] * (block_size - len(content_toks))
                else:
                    padding = []

                token_type_ids = (
                    sum(
                        [
                            [1 if hist["from_sender"] else 0] * len(tok)
                            for tok, hist in zip(hist_toks, history)
                        ],
                        [],
                    )
                    + [1] * len(cand_tok)
                    + [0] * len(padding)
                )

                if i_cand == gold_index and not dummy:
                    label_ids = (
                        [-100] * len(hist_tok) + cand_tok + [-100] * len(padding)
                    )
                else:
                    label_ids = [-100] * block_size

                assert len(content_toks + padding) == block_size
                assert len(token_type_ids) == block_size
                assert len(label_ids) == block_size

                dataset["input_ids"].append(content_toks + padding)
                dataset["token_type_ids"].append(token_type_ids)
                dataset["mc_token_ids"].append(len(content_toks) - 1)
                dataset["label_ids"].append(label_ids)

            dataset["mc_labels"].append(gold_index)
            dataset["overflow"] = False

        for k, v in dataset.items():
            outputs_data[k].append(v)

    for k, v in outputs_data.items():
        assert len(v) == len(
            reply_batch
        ), f"output size mismatch, {len(v)} != {len(reply_batch)}"

    return outputs_data

#!/usr/bin/env python

import json
import os
import random
import sys
import string
from collections import deque
from dataclasses import dataclass, field
from itertools import tee, count
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import tqdm
from transformers import HfArgumentParser, set_seed


@dataclass
class DatasetArguments:
    messages_path: str = field()
    output_file: str = field()
    sender_name: str = field()
    num_distractor: Optional[int] = field(default=1)
    num_history: Optional[int] = field(default=4)
    num_process: Optional[int] = field(default=None)
    seed: Optional[int] = field(default=None)
    ignore_message_requests: bool = field(default=False)
    ignore_archived_threads: bool = field(default=False)
    indent: Optional[int] = field(default=None)
    context_threshold: Optional[int] = field(default=None)
    context_percentile: Optional[float] = field(default=None)
    ascii_threshold: Optional[float] = field(default=0.5)

    def __post_init__(self):
        if self.context_percentile and self.context_threshold:
            raise ValueError(
                "--context_percentile and --context_threshold can't be set at the same time."
            )


def valid_message(message: Dict[str, str]):
    if message["type"] in {
        # "Share",
        "Unsubscribe",
        # "Generic",
        "Call",
        "Subscribe",
        "Payment",
    }:
        return False

    content = message.get("content")
    sender_name = message.get("sender_name")
    if not content or not sender_name:
        return False

    if message["type"] == "Share":
        share_text = message.get("share", {}).get("share_text", "")

        if any(
            match in content
            for match in (
                "8 Ball Pool",
                "Chess:",
                "Robinhood",
                "Words With Friends",
                "waved",
            )
        ):
            return False

        if share_text.startswith("Last update"):
            return False

    return True


def modify_message(message: Dict[str, str]):
    message = message.copy()

    content = message.get("content", "").encode("ascii", "ignore").decode().strip()
    content = content.replace(os.linesep, " ")

    if message["type"] == "Share":
        link = message.get("share", {}).get("link", "")
        share_text = message.get("share", {}).get("share_text", "")

        if not content:
            content = link

        if content.endswith("sent a location."):
            content = share_text
        elif content.endswith("sent a link."):
            content = link
        elif content.endswith("sent an attachment."):
            content = link
        elif link not in content:
            content += " " + link

    if "content" in message:
        message["original_content"] = message["content"]
    message["content"] = content

    return message


def group(chat: Iterable[Dict[str, str]], ms_threshold: int):
    msg_buf = dict(sender_name=None, content=None, timestamp_ms=None)

    for msg in chat:
        sender_name, content, ts = (
            msg["sender_name"],
            msg["content"],
            msg["timestamp_ms"],
        )

        if not sender_name or not content:
            continue

        if msg_buf["sender_name"] != sender_name or (
            ts - msg_buf["timestamp_ms"] >= ms_threshold
        ):
            if msg_buf["sender_name"]:
                yield {
                    "sender_name": msg_buf["sender_name"],
                    "content": msg_buf["content"],
                    "timestamp_ms": ts,
                }

            msg_buf = {
                "sender_name": sender_name,
                "content": content,
                "timestamp_ms": ts,
            }
        else:
            msg_buf["content"] += " " + content

    if msg_buf:
        yield {
            "sender_name": msg_buf["sender_name"],
            "content": msg_buf["content"],
            "timestamp_ms": msg_buf["timestamp_ms"],
        }


def window(
    seq: Iterable[Dict[str, str]], window_size: int, sender_name: str, ms_threshold: int
):
    it = iter(seq)
    win = deque(maxlen=window_size)

    for message in it:
        current_message_ts = message["timestamp_ms"]

        if len(win) > 0:
            last_message_ts = win[-1]["timestamp_ms"]
        else:
            last_message_ts = current_message_ts

        if current_message_ts - last_message_ts >= ms_threshold:
            win.clear()

        win.append(message)

        if len(win) > 1 and win[-1]["sender_name"] == sender_name:
            yield list(win)


def sliding_window(iterable: Iterable, n: int = 2):
    iterables = tee(iterable, n)

    for iterable, num_skipped in zip(iterables, count()):
        for _ in range(num_skipped):
            next(iterable, None)

    return zip(*iterables)


@dataclass
class MessageProcess:
    args: DatasetArguments

    def __call__(self, filepath: str):
        chatlog = json.load(open(filepath))

        all_messages = chatlog["messages"]

        all_messages_length = sum(len(msg.get("content", "")) for msg in all_messages)
        all_printable_length = sum(
            1.0
            for _ in filter(
                lambda x: x in string.printable,
                (s for msg in all_messages for s in msg.get("content", "")),
            )
        )

        if (
            all_messages_length == 0
            or all_printable_length / all_messages_length <= self.args.ascii_threshold
        ):
            return []

        all_messages = map(modify_message, all_messages)
        all_messages = filter(valid_message, all_messages)
        all_messages = sorted(all_messages, key=lambda x: x["timestamp_ms"])

        if len(all_messages) < 2:
            return []

        time_threshold = float("inf")
        if self.args.context_threshold:
            time_threshold = self.args.context_threshold
        elif len(all_messages) >= 2 and self.args.context_percentile:
            time_threshold = np.percentile(
                [
                    t2["timestamp_ms"] - t1["timestamp_ms"]
                    for t1, t2 in sliding_window(all_messages)
                ],
                self.args.context_percentile,
            )

        sent_messages, all_messages = tee(group(all_messages, time_threshold))

        sent_messages = set(
            msg["content"]
            for msg in filter(
                lambda m: m["sender_name"] == self.args.sender_name,
                sent_messages,
            )
        )

        if len(sent_messages) <= self.args.num_distractor + 1:
            return []

        examples = []

        for log in window(
            all_messages,
            self.args.num_history + 1,
            self.args.sender_name,
            time_threshold,
        ):
            chatlog = [
                {
                    "from_sender": msg["sender_name"] == self.args.sender_name,
                    "content": msg["content"],
                }
                for msg in log
            ]
            example = {
                "history": chatlog[:-1],
                "reply": chatlog[-1]["content"],
                "distractor": random.sample(
                    set(
                        filter(
                            lambda s: s != chatlog[-1]["content"],
                            sent_messages,
                        )
                    ),
                    self.args.num_distractor,
                ),
            }
            examples.append(example)

        return examples


def main():
    parser = HfArgumentParser(DatasetArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (data_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (data_args,) = parser.parse_args_into_dataclasses()

    data_args: DatasetArguments

    if data_args.seed:
        set_seed(data_args.seed)

    messages_path = []
    for root, dirs, files in os.walk(data_args.messages_path):
        if data_args.ignore_message_requests and "message_requests" in root:
            continue

        if data_args.ignore_archived_threads and "archived_threads" in root:
            continue

        for name in files:
            if name.startswith("message_") and name.endswith((".json")):
                full_path = os.path.join(root, name)
                messages_path.append(full_path)

    if data_args.num_process is None:
        data_args.num_process = cpu_count() - 1

    Path(os.path.dirname(data_args.output_file)).mkdir(parents=True, exist_ok=True)

    with Pool(data_args.num_process) as pool, open(data_args.output_file, "w") as f:
        f.truncate(0)

        process = MessageProcess(args=data_args)

        pbar = tqdm.tqdm(total=len(messages_path))
        for res in pool.imap_unordered(process, messages_path):
            if res:
                f.write(
                    os.linesep.join(
                        json.dumps(s, sort_keys=True, indent=data_args.indent)
                        for s in res
                    )
                )
            pbar.update()


if __name__ == "__main__":
    main()

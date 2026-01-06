#!/usr/bin/env python3

# ╭──────────────────────────────────────╮
# │ main.py on Nelm                      │
# │ Nercone <nercone@diamondgotcat.net>  │
# │ Made by Nercone / MIT License        │
# │ Copyright (c) 2025 DiamondGotCat     │
# ╰──────────────────────────────────────╯

import json
import torch
from os import listdir
from os.path import isfile, join
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset
from nercone_modern.logging import ModernLogging
from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM

VERSION = "1.0.0"

class NelmTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length=8192):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index) -> dict:
        text = self.texts[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class NelmTextFileDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length=8192):
        self.path = Path(path)
        self.files = [f for f in listdir(str(self.path)) if isfile(join(str(self.path), f))]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index) -> dict:
        path = self.files[index]
        with open(path, "r") as f:
            text = f.read()
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class NelmTalkDataset(Dataset):
    def __init__(self, conversations: list[list[dict[str,str]]], tokenizer, max_length=8192):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index) -> dict:
        text = self.build_text(self.conversations[index])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def build_text(self, conversation: list[dict[str,str]]):
        result = ""
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            result += f"<|start|>{role}<|message|>{content}<|end|>"
        return result

def int_prompt(logger: ModernLogging, message: str = "", default: int | None = None) -> int:
    while True:
        result = logger.prompt(message, default=str(default)).strip()
        try:
            result = int(result)
            return result
        except:
            logger.log("整数のみが利用可能です。")

def float_prompt(logger: ModernLogging, message: str = "", default: float | None = None) -> float:
    while True:
        result = logger.prompt(message, default=str(default)).strip()
        try:
            result = float(result)
            return result
        except:
            logger.log("少数のみが利用可能です。")

def main() -> int:
    logger = ModernLogging("nelm-builder", display_level="INFO")
    logger.log(f"Nelm Builder {VERSION}")

    logger.log("Nelm Builderへようこそ。まず初めに、ベースにするモデルを指定してください。")
    model_id = logger.prompt("ベースモデル", default="Zeta-DGC/Zeta-2", show_default=True)
    logger.log("次に、ベースモデルの利用範囲を指定してください。")
    logger.log("重みまで全て使用する場合は'all'を、トークナイザや設定などのみを使用する場合は'min'を選択してください。")
    usage_level = logger.prompt("モデルの利用範囲", default="min", choices=["all", "min"], show_default=True)
    if usage_level == "all":
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif usage_level == "min":
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.log("ベースモデルの設定が完了しました。次はデータセットを設定します。")
    logger.log("まず、コンテキスト長を設定します。これはモデルの生成できる最大の応答サイズになります。")
    max_length = int_prompt(logger, "コンテキスト長", default=8192)
    logger.log("次にデータセット本体を読み込みます。")
    logger.log("データセットの形式には、複数のテキストファイルが入ったディレクトリ(textfiles)、AzukiFormat(AzukiF)形式のJSONファイル、またはHuggingFaceリポジトリが利用可能です。")
    dataset_type = logger.prompt("どれを使用しますか？", default="azukif", choices=["textfile", "azukif", "huggingface"], show_default=True)
    if dataset_type == "textfile":
        logger.log("ディレクトリパスを指定してください。")
        while True:
            _dataset_path = logger.prompt("パス").strip()
            if Path(_dataset_path).is_dir():
                dataset_path = _dataset_path
                break
            else:
                logger.log("ディレクトリが見つかりませんでした。打ち間違えがないか確認してください。")
        logger.log("データセットを読み込みます。これには時間がかかる場合があります。")
        dataset = NelmTextFileDataset(path=dataset_path, tokenizer=tokenizer, max_length=max_length)
        logger.log("お待たせしました。データセットの読み込みが完了しました。")
    elif dataset_type == "azukif":
        logger.log("ファイルパスを指定してください。")
        while True:
            _dataset_path = logger.prompt("パス").strip()
            if Path(_dataset_path).is_file():
                dataset_path = _dataset_path
                break
            else:
                logger.log("ファイルが見つかりませんでした。打ち間違えがないか確認してください。")
        logger.log("データセットを読み込みます。これには時間がかかる場合があります。")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset = NelmTalkDataset(conversations=data, tokenizer=tokenizer, max_length=max_length)
        logger.log("お待たせしました。データセットの読み込みが完了しました。")
    elif dataset_type == "huggingface":
        logger.log("HuggingFaceのリポジトリのパスを指定してください。これは通常'ユーザー名/データセット名'のような形式です。")
        dataset_path = logger.prompt("パス", default="Zeta-DGC/Zeta-2-Dataset", show_default=True).strip()
        logger.log("サブセット名を指定してください。サブセット名がない場合、何も書かずにエンターキーを押してください。")
        subset_name = logger.prompt("パス", default="").strip()
        if subset_name.strip() != "":
            raw_dataset = load_dataset(dataset_path, subset_name)
        else:
            raw_dataset = load_dataset(dataset_path)
        logger.log("学習に使用するデータセット名を指定してください。大抵の場合、これは'train'です。")
        dataset_name = logger.prompt("データセット名", default="train").strip()
        column_list = list(raw_dataset[dataset_name].features.keys())
        logger.log(f"利用可能なカラム: {', '.join(column_list)}")
        user_column = logger.prompt("userロールに使用するカラムはどれですか？", choices=column_list)
        assistant_column = logger.prompt("assistantロールに使用するカラムはどれですか？", choices=column_list)
        logger.log("データセットを読み込みます。これには時間がかかる場合があります。")
        conversations: list[list[dict[str,str]]] = []
        for turn in raw_dataset[dataset_name]:
            conversations.append([{"role": "user", "content": turn[user_column]}, {"role": "assistant", "content": turn[assistant_column]}])
        dataset = NelmTalkDataset(conversations=conversations, tokenizer=tokenizer, max_length=max_length)
        logger.log("お待たせしました。データセットの読み込みが完了しました。")

    device = "cpu"
    use_cuda = False
    float_mode = "none"
    if torch.cuda.is_available():
        logger.log("データセットの設定が完了しました。次に、高速化の設定を行います。")
        logger.log("まず、CUDAの使用についてです。NVIDIA製GPUを搭載している場合、CUDAを使用して高速に処理することができます。")
        use_cuda = logger.prompt("CUDAを使用しますか？", default="Y", choices=["Y", "n"], show_default=True) == "Y"
        if use_cuda:
            device = "cuda"
        else:
            device = "cpu"
        model.to(device)
        if use_cuda and torch.cuda.is_bf16_supported():
            logger.log("fp16やbf16では、精度を落とす代わりに、ある程度の高速化が可能です。")
            logger.log("これにはfp16/bf16の2種類がありますが、GPUによって対応状況が異なります。")
            logger.log("この機能を利用しますか？また、どちらを使用しますか？")
            float_mode = logger.prompt("", default="none", choices=["none", "fp16", "bf16"], show_default=True)
        else:
            logger.log("fp16では、精度を落とす代わりに、ある程度の高速化が可能です。")
            logger.log("この機能を利用しますか？")
            float_mode = "fp16" if logger.prompt("", default="N", choices=["y", "N"], show_default=True) == "y" else "none"
        logger.log("高速化の設定が完了しました。次に、学習に関する設定を行います。")
    else:
        logger.log("データセットの設定が完了しました。次に、学習に関する設定を行います。")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=int_prompt(logger, "num_train_epochs", default=3),
        per_device_train_batch_size=int_prompt(logger, "per_device_train_batch_size", default=1),
        gradient_accumulation_steps=int_prompt(logger, "gradient_accumulation_steps", default=4),
        learning_rate=float_prompt(logger, "learning_rate", default=3e-5),
        warmup_steps=int_prompt(logger, "warmup_steps", default=100),
        weight_decay=float_prompt(logger, "weight_decay", default=0.01),
        logging_dir="./logs",
        logging_steps=int_prompt(logger, "logging_steps", default=10),
        save_strategy="no",
        fp16=float_mode=="fp16",
        bf16=float_mode=="bf16"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    logger.log("学習に関する設定が完了しました。学習を開始します。")
    float_mode = logger.prompt("準備はいいですか？", default="Y", choices=["Y", "n"], show_default=True)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.log("中断されました。", "WARNING")
        export_path = logger.prompt("途中経過のモデルをどこに保存しますか？", default="./interrupted_model/", show_default=True)
        model.save_pretrained(export_path)
        tokenizer.save_pretrained(export_path)
        logger.log("またお会いしましょう！")
        return 130
    logger.log("学習が完了しました！")
    export_path = logger.prompt("どこに保存しますか？", default="./your_model/", show_default=True)
    model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)
    logger.log("またお会いしましょう！")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

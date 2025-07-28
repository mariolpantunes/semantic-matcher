import pathlib
import glob
import time

import pandas as pd
import numpy as np

from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

from datasets import load_dataset



class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)


class Sbert_model():

    def __init__(self, corpus_path, vector_size, output, pretrained="from_scratch", semantic_training=True) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)

        if pretrained == "pretrained":
            self.model = SentenceTransformer("roberta-base")

            if semantic_training:
                self.train_model(str(self.output / "base_model"), str(self.output / "trained_model"))

        elif pretrained == "pretrained_optimized":

            dataset = pathlib.Path(corpus_path)
            preprocessed_train_dataset = dataset/"processed_train_setences.txt"
            preprocessed_test_dataset = dataset/"processed_train_setences.txt"

            pre_process_time = self.dataset_preprocessing(dataset, preprocessed_train_dataset, preprocessed_test_dataset)


            per_device_train_batch_size = 64
            save_steps = 1000               #Save model every 1k steps
            num_train_epochs = 3           #Number of epochs
            use_fp16 = True                #Set to True, if your GPU supports FP16 operations
            max_length = 100                #Max length for a text input
            mlm_prob = 0.15                 #Probability that a word is replaced by a [MASK] token

            model = AutoModelForMaskedLM.from_pretrained("roberta-base")
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")

            ##### Load our training datasets

            train_sentences = []
            with open(preprocessed_train_dataset, 'r', encoding='utf8') as fIn:
                for line in fIn:
                    line = line.strip()
                    if len(line) >= 10:
                        train_sentences.append(line)
                        
            dev_sentences = []
            with open(preprocessed_test_dataset, 'r', encoding='utf8') as fIn:
                for line in fIn:
                    line = line.strip()
                    if len(line) >= 10:
                        dev_sentences.append(line)

            train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
            dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None

            ##### Training arguments

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

            training_args = TrainingArguments(
                output_dir= str(self.output / "base_model" / "checkpoints"),
                overwrite_output_dir=True,
                num_train_epochs=num_train_epochs,
                eval_strategy="steps" if dev_dataset is not None else "no",
                per_device_train_batch_size=per_device_train_batch_size,
                eval_steps=save_steps,
                save_steps=save_steps,
                logging_steps=save_steps,
                save_total_limit=1,
                prediction_loss_only=True,
                fp16=use_fp16
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset
            )

            tokenizer.save_pretrained(str(self.output / "base_model"))
            #trainer.train()
            model.save_pretrained(str(self.output / "base_model"))

            self.model = SentenceTransformer(str(self.output / "base_model"))
            if semantic_training:
                self.train_model(str(self.output / "base_model"), str(self.output / "trained_model"))
        else:
            dataset = pathlib.Path(corpus_path)
            preprocessed_train_dataset = dataset/"processed__train_setences.txt"
            preprocessed_test_dataset = dataset/"processed__train_setences.txt"

            (self.output).mkdir(parents=True, exist_ok=True)

            per_device_train_batch_size = 64

            save_steps = 1000               #Save model every 1k steps
            num_train_epochs = 3            #Number of epochs
            use_fp16 = True                #Set to True, if your GPU supports FP16 operations
            max_length = 100                #Max length for a text input
            mlm_prob = 0.15                 #Probability that a word is replaced by a [MASK] token

            pre_process_time = self.dataset_preprocessing(dataset, preprocessed_train_dataset, preprocessed_test_dataset)

            ### Train tokenizer from scratch
            train_files = glob.glob(str(dataset)+'/*.csv')

            tokenizer = ByteLevelBPETokenizer()

            tokenizer.train(files=train_files, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

            tokenizer.save_model(str(self.output))

            #####################################

            #### Initialize model and tokenizer for pretended task
            config = RobertaConfig(
                max_position_embeddings=514,
                hidden_size=vector_size,
                num_attention_heads=12,
                num_hidden_layers=6,
                type_vocab_size=1
            )

            tokenizer = RobertaTokenizerFast.from_pretrained(str(self.output), max_length=512)

            model = RobertaForMaskedLM(config=config)

            train_sentences = []
            with open(preprocessed_train_dataset, 'r', encoding='utf8') as fIn:
                for line in fIn:
                    line = line.strip()
                    if len(line) >= 10:
                        train_sentences.append(line)

            dev_sentences = []
            with open(preprocessed_test_dataset, 'r', encoding='utf8') as fIn:
                for line in fIn:
                    line = line.strip()
                    if len(line) >= 10:
                        dev_sentences.append(line)

            train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
            dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

            training_args = TrainingArguments(
                output_dir=str(self.output/ "base_model" / "checkpoints"),
                overwrite_output_dir=True,
                num_train_epochs=num_train_epochs,
                eval_strategy="steps" if dev_dataset is not None else "no",
                per_device_train_batch_size=per_device_train_batch_size,
                eval_steps=save_steps,
                save_steps=save_steps,
                logging_steps=save_steps,
                save_total_limit=1,
                prediction_loss_only=True,
                fp16=use_fp16
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset
            )

            tokenizer.save_pretrained(str(self.output))
            trainer.train()
            model.save_pretrained(str(self.output / "base_model"))

            self.model = SentenceTransformer(str(self.output / "base_model"))

            if semantic_training:
                self.train_model(str(self.output / "base_model"), str(self.output / "trained_model"))

    def train_model(self, base_model_path, trained_model_path):

        #Get train examples
        train_dataset = load_dataset("sentence-transformers/stsb", split="train")
        eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
        test_dataset = load_dataset("sentence-transformers/stsb", split="test")

        train_loss = losses.CosineSimilarityLoss(self.model)

        dev_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-dev",
        )

        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir= trained_model_path + "/checkpoints",
            # Optional training parameters:
            num_train_epochs=3,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            logging_steps=1000,
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=dev_evaluator,
        )

        trainer.train()

        self.model.save_pretrained(trained_model_path)

    def dataset_preprocessing(self, dataset, preprocessed_train_dataset, preprocessed_test_dataset):
        train_files = glob.glob(str(dataset)+'/*.csv')

        # Read the files in the dataset and create setences
        print('Generating tokens from files.')
        # Text Mining Pipeline
        aggregated_files = open(preprocessed_train_dataset, "w")

        for f in train_files[0:int(len(train_files)*0.8)]:
            with open(f, 'rt', newline='', encoding='utf-8') as f:
                snippets = f.readlines()
                for s in snippets:
                    aggregated_files.write(s)

        dev_files = open(preprocessed_test_dataset, "w")

        for f in train_files[int(len(train_files)*0.8):]:
            with open(f, 'rt', newline='', encoding='utf-8') as f:
                snippets = f.readlines()
                for s in snippets:
                    dev_files.write(s)

        aggregated_files.close()

        dev_files.close()

    def fit(self, text):
        pass

    def predict(self, x, y):
        term_1 = self.model.encode(x,show_progress_bar=False)
        term_2 = self.model.encode(y,show_progress_bar=False)
        return np.dot(term_1, term_2)/(np.linalg.norm(term_1)*np.linalg.norm(term_2))
        
    def calculate_bias(self, list_words):
        pass
# Importing libraries
import os
from argparse import ArgumentParser
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

from rich.table import Column, Table
from rich import box
from rich.console import Console
# Import cross-project dependencies
from readeer import DataReader
# Setting up the device for GPU usage
from torch import cuda
from accelerate import Accelerator
accelerator = Accelerator()
device = cuda


# device = 'cuda' if cuda.is_available() else 'cpu'
# define a rich console logger
console = Console(record=True)

model_params = {
    "MODEL": "t5-small",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 1,  # training batch size
    "VALID_BATCH_SIZE": 1,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 500,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 500,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}


def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(Column("source_text", justify="center"), Column("target_text", justify="center"), title="Sample Data",
                  pad_edge=False, box=box.ASCII)

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)


training_logger = Table(Column("Epoch", justify="center"),
                        Column("Steps", justify="center"),
                        Column("Loss", justify="center"),
                        title="Training Status", pad_edge=False, box=box.ASCII)


def train(epoch, tokenizer, model, device, loader, optimizer):
    """
  Function to be called for training with the parameters passed from main function
  """
    model.train()
    for _, data in tqdm(enumerate(loader, 0), disable=not accelerator.is_local_main_process):
        y = data['target_ids']
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids']
        mask = data['source_mask']

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if _ % 100 == 0:
        #     training_logger.add_row(str(epoch), str(_), str(loss))
            accelerator.print(f"Epoch: {epoch}, Loss: {loss}")

        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()


def validate(tokenizer, model, device, loader):
    """
  Function to evaluate model for predictions
  """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(accelerator.device, dtype=torch.long)
            ids = data['source_ids'].to(accelerator.device, dtype=torch.long)
            mask = data['source_mask'].to(accelerator.device, dtype=torch.long)
            #generated_ids = model.module.generate(
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=500,
                num_beams=6,
                num_return_sequences = 2,
                repetition_penalty=1.5,
                length_penalty=2.5,
                early_stopping=False,
                pad_token_id=tokenizer.eos_token_id
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 100 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def T5Trainer(dataframe, source_text, target_text, model_params, output_dir="./outputs/"):
    """
    T5 trainer
    """

    # Set random seeds and deterministic pytorch for reproducibility
    # torch.manual_seed(model_params["SEED"])  # pytorch random seed
    # np.random.seed(model_params["SEED"])  # numpy random seed
    # torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    # tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    # model = AutoModelForCausalLM.from_pretrained(model_params["MODEL"])
    # model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    # display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size)
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = DataReader(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                              model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = DataReader(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                         model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        # 'num_workers': 0
    }

    val_params = {
        'batch_size': model_params["VALID_BATCH_SIZE"],
        'shuffle': False,
        # 'num_workers': 0
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_params["LEARNING_RATE"])

    model, optimizer, training_loader = accelerator.prepare(model, optimizer, training_loader)

    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    accelerator.wait_for_everyone()
    path = os.path.join(output_dir, "model_files")
    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv(os.path.join(output_dir, 'predictions.csv'))

    console.save_text(os.path.join(output_dir, 'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-i', '--input', type=str)
    ap.add_argument('-o', '--output', type=str, default='outputs')
    ap.add_argument('-cv', '--console-verbosity', default='info', help='Console logging verbosity')
    ap.add_argument('-fv', '--file-verbosity', default='debug', help='File logging verbosity')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--sharded_ddp')
    args = ap.parse_args()

    df = pd.read_csv("csv_file.csv")
    #df["user"] = "input: " + df["user"]
    # df["system"] = "output: " + df["system"]
    T5Trainer(
        dataframe=df.sample(n=50) if args.test else df,
        source_text="system",
        target_text="user",
        model_params=model_params,
        output_dir=args.output
    )
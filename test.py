from argparse import ArgumentParser

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

model_params = {
    "MODEL": "t5-small",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 1,  # training batch size
    "VALID_BATCH_SIZE": 1,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}
def test(text):
    source = tokenizer.batch_encode_plus([text], max_length=512, pad_to_max_length=True,
                                         truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].to(dtype=torch.long)
    source_mask = source['attention_mask'].to(dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=15,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences = 2
        )
        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                 generated_ids]
    return preds


ap = ArgumentParser()
ap.add_argument('--model', type=str)
args = ap.parse_args()

tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
model.load_state_dict(torch.load('outputs/model_files/pytorch_model.bin'))
model.eval()
while True:
    # result = test("how are you?")
    result = test(input(">> User: "))
    print(f"Alana: {result}")
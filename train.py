from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# for now, use opus_books for demo
# hopefully transition to better dataset later on
    # kaggle datasets download -d ramakrishnan1984/785-million-language-translation-database-ai-ml
# can also try to set up individual config files that specify model and dataset like in mdl (maybe overkill)
source_lang = 'en'
target_lang = 'de'
prefix = "translate English to German: "

def preprocess(examples):
    # gets the english (inputs) part of each translation
    inputs = [prefix + example[source_lang] for example in examples['translation']]
    # gets the german (target) part of each translation
    targets = [example[target_lang] for example in examples['translation']]

    # turns the inputs into tokens (vectors? not sure)
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

def postprocess_text(preds, labels):
    # preds == predictions, probably outputs
    # labels are the word embeddings? not sure
    preds = [pred.strip() for pred in preds] # no clue what this means
    labels = [[label.strip()] for label in labels] # no clue what this means
    return preds, labels

def compute_metrics(eval_preds): # output from the model?
    preds, labels = eval_preds 
    if isinstance(preds, tuple):
        preds = preds[0]
    # tokenizer converts predictions into somethin else idk man
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # postprocess the guys
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # calculate the result using the libraries
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {'bleu': result['score']}

    # idk whats happening anymore :(
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result['gen_len'] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if __name__ == '__main__':
    # load dataset
    dataset = load_dataset('opus_books', 'de-en')
    
    # get training split
    dataset = dataset['train'].train_test_split(test_size=0.2)

    # t5 tokenizer processes the language pairs
    checkpoint = 'google-t5/t5-small'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # use datasets map method to preprocess the dataset
    tokenized_dataset = dataset.map(preprocess, batched=True)
    # groups the examples into batches
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    # the metric used to check loss?
    metric = evaluate.load('sacrebleu')

    # get the model from the pretrained checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="my_awesome_opus_books_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )

    # get the training thing???
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
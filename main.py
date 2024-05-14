from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

if __name__ == '__main__':
    # gets the model
    tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")

    # old way, goes through hub
    # translator = pipeline("translation_de_to_en", model='my_awesome_opus_books_model') # before renaming
    # translator = pipeline("translation_de_to_en", model='t5_opus_books_de_en_model') # after rename
    
    # prefix for translation
    text = "translate German to English: "
    # text = "translate English to German: "

    # the actual text to be translated
    # text += "Bitteschön, Auf Wiedersehen!"
    text += "Es war ganz unmöglich, an diesem Tage einen Spaziergang zu machen."

    # tokenizes the text
    inputs = tokenizer(text, return_tensors="pt").input_ids

    # generates the outputs using the model
    model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
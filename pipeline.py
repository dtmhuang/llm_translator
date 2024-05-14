from transformers import pipeline
if __name__ == '__main__':
    translator = pipeline("translation_de_to_en", model='my_awesome_opus_books_model') # before renaming
    # translator = pipeline("translation_de_to_en", model='t5_opus_books_de_en_model') # after rename
    text = "translate German to English: "

    text += "Bitteschön, Auf Wiedersehen!"
    translator(text)
    print(text)
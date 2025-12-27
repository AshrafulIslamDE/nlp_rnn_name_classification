import string

import torch
from unidecode import unidecode

allowed_characters = string.ascii_letters + " .,;'" + "_"
nums_letters=len(allowed_characters)

def convert_unicode_to_ascii(text:str)-> str :
    return unidecode(text)

def letter_to_index(letter:str)-> int:
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    return allowed_characters.index(letter)

def transform_text_to_tensor(text:str)-> torch.Tensor :
     converted_text = convert_unicode_to_ascii(text)
     text_length=len(converted_text)
     text_tensor=torch.zeros(text_length,nums_letters)

     # one hot coding
     for i,letter in enumerate(converted_text):
         text_tensor[i][letter_to_index(letter)]=1

     return text_tensor

if __name__ == "__main__":
    tensor=transform_text_to_tensor("hello world")
    print(tensor.shape)


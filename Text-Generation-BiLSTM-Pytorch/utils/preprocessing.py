import numpy as np
import torch
class Preprocessing:

    @staticmethod
    def read_dataset(file):

        letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
					'n','o','p','q','r','s','t','u','v','w','x','y','z',' ', ",", ".", "“", "”", "?", "’"]

        with open(file, "r", encoding="utf-8") as f:
            raw_text=f.readlines()

        raw_text=[line.lower() for line in raw_text]

        # Create a string which contains the entire text
        text_string = ''
        for line in raw_text:
            text_string+=line.strip()

        # Create an array by characters(char)
        text=list()

        for char in text_string:
            text.append(char)

        text=[char for char in text_string if char in letters]

        return text

    
    @staticmethod
    def create_dictionary(text):
        char_to_idx = dict()
        idx_to_char = dict()
		
        idx = 0

        for char in text:
            if char not in char_to_idx.keys():
                char_to_idx[char]=idx
                idx_to_char[idx]=char
                idx +=1

        print("Vocab:", len(char_to_idx))

        return char_to_idx, idx_to_char
    
    @staticmethod
    def build_sequence_target(text, char_to_idx, window):

        x=list()
        y=list()

        for i in range(len(text)):
            try:
                sequence=text[i:i+window]
                sequence=[char_to_idx[char] for char in sequence]

                target=text[i+window]
                target=char_to_idx[target]

                x.append(sequence)
                y.append(target)

            except:
                pass

        x=np.array(x)
        y=np.array(y)

        return x, y


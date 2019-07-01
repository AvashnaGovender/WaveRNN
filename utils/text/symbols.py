""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from utils.text import cmudict

_pad = '#'
_special = '@'
_letters = 'ACDEGHIJLMNOSTUZabdefghiklmnoprstuvwyz'
_numbers = '0123456'
_word_boundary = '|'


# Export all symbols:
symbols = list(_pad) + list(_letters)+ list(_numbers)

import numpy as np

def get_text_features(data): 
    
    '''
        Args:
            str, input data
            
        Returns: 
            np.array, shape=(22,);
            an array of the text converted to features
            
    '''
    
    assert type(data) == str, f'Expected type {str}. Received {type(data)}.'

    n_upper = 0
    n_lower = 0
    n_alpha = 0
    n_digits = 0
    n_spaces = 0
    n_numeric = 0
    n_special = 0
    number = 0
    special_chars = {'&': 0, '@': 1, '#': 2, '(': 3, ')': 4, '-': 5, '+': 6, 
                    '=': 7, '*': 8, '%': 9, '.':10, ',': 11, '\\': 12,'/': 13, 
                    '|': 14, ':': 15}
    
    special_chars_arr = np.zeros(shape=len(special_chars))    
    
    # character wise
    for char in data: 
          
        # for lower letters 
        if char.islower(): 
            n_lower += 1
  
        # for upper letters 
        if char.isupper(): 
            n_upper += 1
        
        # for white spaces
        if char.isspace():
            n_spaces += 1
        
        # for alphabetic chars
        if char.isalpha():
            n_alpha += 1
        
        # for numeric chars
        if char.isnumeric():
            n_numeric += 1
        
        # array for special chars
        if char in special_chars.keys():
            char_idx = special_chars[char]
            # put 1 at index
            special_chars_arr[char_idx] += 1
            
    # word wise
    for word in data.split():
        
        # if digit is integer 
        try:
            number = int(word)
            n_digits += 1
        except:
            pass

        # if digit is float
        if n_digits == 0:
            try:
                number = float(word)
                n_digits += 1
            except:
                pass
    
    features = []
    features.append([n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_digits])
    features = np.array(features)
    features = np.append(features, np.array(special_chars_arr))
    
    return features

# if __name__ == "__main__":
#     STRING = 'aA 12@#%&*'
#     print(get_text_features(STRING))
# Importing SequenceMatcher
# from difflib module
from difflib import SequenceMatcher

# Declaring string variables
str1 = 'hello'
str2 = 'hello'

# Using the SequenceMatcher()
match = SequenceMatcher(None,str1, str2)

# convert above output into ratio
# and multiplying it with 100
res = match.ratio() * 100

# Display the final result
print(int(res), "%")

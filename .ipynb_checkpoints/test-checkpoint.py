### to test out function decorators
import functools
def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(word):
        return word 
    return wrapper_decorator

@decorator
def test(word):
    print('hi')

print(test('abcd'))

# to test out regex
import re
A = 'C#'

import random
import string

def generate_password(length=8):
    if length < 4:
        raise ValueError("Password length must be at least 4 characters.")

    all_characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(all_characters) for _ in range(length))

if __name__ == '__main__':
    print(generate_password())
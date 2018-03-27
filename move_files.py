import os
from random import randint

print(os.getcwd())

os.chdir('data/like_dislike/train/dislike')
print(os.getcwd())

filenames = os.listdir()
print(filenames)

num = 88 

folder = "dislike_test/"

os.rename(filenames[0], folder + filenames[0])
print(os.getcwd())


for i in range(num):
    index = randint(0, len(filenames))
    os.rename(filenames[index], folder + filenames[index])
    print("moved: " + str(filenames[index]))
    
    print(len(filenames))
    del(filenames[index])
    print(len(filenames))






























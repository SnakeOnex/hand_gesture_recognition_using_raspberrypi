import os
from random import randint

print(os.getcwd())

os.chdir('data2/dislike/')
print(os.getcwd())

filenames = os.listdir()
print(filenames)

num = 99 

folder = "test/"

os.rename(filenames[0], folder + filenames[0])
print(os.getcwd())


for i in range(num):
    index = randint(0, len(filenames))
    os.rename(filenames[index], folder + filenames[index])
    print(f"moved filenames[index]")
    
    print(len(filenames))
    del(filenames[index])
    print(len(filenames))






























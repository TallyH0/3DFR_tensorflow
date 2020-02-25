import os

def mkdir(dname):
  if not os.path.exists(dname):
    os.makedirs(dname)
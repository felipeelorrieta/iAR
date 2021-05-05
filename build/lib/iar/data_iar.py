import pandas as pd

def clcep():
    return pd.read_csv("../data/clcep.csv", encoding='latin-1',header=0,sep=",")

def eb():
    return pd.read_csv("../data/eb.csv", encoding='latin-1',header=0,sep=",")

def dmcep():
    return pd.read_csv("../data/dmcep.csv", encoding='latin-1',header=0,sep=",")

def dscut():
    return pd.read_csv("../data/dscut.csv", encoding='latin-1',header=0,sep=",")

def Planets():
    return pd.read_csv("../data/Planets.csv", encoding='latin-1',header=0,sep=",")

def agn():
    return pd.read_csv("../data/agn.csv", encoding='latin-1',header=0,sep=",")

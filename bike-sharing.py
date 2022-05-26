# importing libraries
import numpy as np
import pandas as pd

# plot libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (16, 6)

#dataset reading
df = pd.read_csv("https://pycourse.s3.amazonaws.com/bike-sharing.csv")

# date time conversion
df['datetime'] = pd.to_datetime(df['datetime'])


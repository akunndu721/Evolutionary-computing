#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# data
data = {
    'Number of Cities': [14, 51, 52, 76, 105, 127, 137, 195, 314, 575],
    'GA (length)': [24.05, 476.41, 7816.51, 655.81, 21914.67, 165636.56, 1346.26, 4580.88, 139304.22, 33414.12],
    'ACO-GA (length)': [25.96, 466.84, 7793.14, 574.2, 16547.12, 129574.17, 867.88, 2770.49, 51284.09, 8844.41],
    'Optimization': [-7.94, 2.01, 0.30, 12.44, 24.49, 21.77, 35.53, 39.52, 63.19, 73.53]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Number of Cities'], df['Optimization'], marker='o', linestyle='-', color='b')
plt.title('Number of Cities vs Optimization')
plt.xlabel('Number of Cities')
plt.ylabel('Optimization')
plt.grid(True)
plt.show()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# data
data = {
    'Number of Cities': [14, 51, 52, 76, 105, 127, 137, 195, 314, 575],
    'GA (length)': [24.05, 476.41, 7816.51, 655.81, 21914.67, 165636.56, 1346.26, 4580.88, 139304.22, 33414.12],
    'ACO-GA (length)': [25.96, 466.84, 7793.14, 574.2, 16547.12, 129574.17, 867.88, 2770.49, 51284.09, 8844.41],
    'Optimization': [-7.94, 2.01, 0.30, 12.44, 24.49, 21.77, 35.53, 39.52, 63.19, 73.53]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Number of Cities'], df['GA (length)'], marker='o', linestyle='-', color='b', label='GA')
plt.plot(df['Number of Cities'], df['ACO-GA (length)'], marker='o', linestyle='-', color='r', label='ACO-GA')
plt.title('Number of Cities vs Path Length')
plt.xlabel('Number of Cities')
plt.ylabel('Path Length')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





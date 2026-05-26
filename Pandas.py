import pandas as pd
data={
  "name":["ram","sita"],
  "Age":[20,21],
  "Marks":[85.5,87]
  {
df=pd.DataFrame(data)
print(df.select_dtype(include='number'))

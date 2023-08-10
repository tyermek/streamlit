import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

data = pd.read_csv('Dataset.csv', parse_dates=['Attribute'],
                   index_col=['Attribute'])
df = data.drop(columns=['Unnamed: 0', 'Year', 'Month', 'Day', 'datetime'])

st.header('Обучение нейронной модели')
# Display the first 5 rows of a Pandas DataFrame

st.write('Table 1 - Фрагмент обущающей выборки')

st.write(df.head())

# Display some graphs (replace with your own code)
st.write('Graph 1 - Корреляционный анализ')
cat = ['Cl','W2', 'tS', "E'"]
for i in cat:
    df[i] = pd.factorize(df[i])[0]
fig = plt.figure(figsize=(10, 4))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.1g')
plt.title("Корреляционный анализ", fontsize=18);
st.pyplot(fig)

st.write('Graph 2 - Фрагмент исходных данных выработки электрической энергии электростанции Жалагаш')
data_subset = data.tail(72)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_subset["datetime"], y=data_subset["Value"], mode="markers", name="Data"))
fig.add_trace(go.Scatter(x=data_subset["datetime"], y=data_subset["Value"], mode="lines", name="Line"))
fig.update_layout(title="Фрагмент исходных данных выработки электрической энергии электростанции Жалагаш",
                  xaxis_title="Дата и время",
                  yaxis_title="Объем выработки, МВт")
st.plotly_chart(fig, use_container_width=True)

st.write('Graph 2 - Фрагмент исходных данных выработки электрической энергии электростанции Жалагаш (30 МВт)')

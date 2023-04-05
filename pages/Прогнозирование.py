import streamlit as st
import numpy as np
import datetime

st.header('Прогнозирование')

import xgboost as xgb
# Load the model
loaded_model = xgb.Booster()


model = st.selectbox('Выберите электростанцию', ('Объект 1', 'Объект 2'))
if model == 'Объект 1':
    loaded_model.load_model('model_zhalagash_1.bin')
elif model == 'Объект 2':
    loaded_model.load_model('model_Shu.bin')
times = []
for hours in range(0, 24):
    times.append(datetime.time(hours, 0))
Time = st.selectbox("Time", times, key="time", format_func=lambda t: t.strftime("%H:%M"))
s = st.selectbox('Солнце взошло?', ('Да', 'Нет'))
if s == 'Да':
    s = 1
else:
    s = 0
sH = st.slider('Длина солнечного дня?', 0, 18, 10)
T = st.number_input('Температура (C)')
p = st.number_input('Количество осадком (мм)')
w = st.selectbox('Тип погоды', ('ясно', 'облачность', 'пасмурно', 'облачно', 'ливневый снег',
                                'слабый снег', 'слабый дождь', 'дождь', 'дымка', 'туман, изморозь',
                                'малооблачно', 'сильный туман', 'гроза', 'пыль, ветер',
                                'слабая морось', 'снег', 'ливневая крупа'))
if w == 'ясно':
    w = 0
elif w == 'облачность':
    w = 1
elif w == 'пасмурно':
    w = 2
elif w == 'облачно':
    w = 3
elif w == 'ливневый снег':
    w = 4
elif w == 'слабый снег':
    w = 5
elif w == 'слабый дождь':
    w = 6
elif w == 'дождь':
    w = 7
elif w == 'дымка':
    w = 8
elif w == 'туман, изморозь':
    w = 9
elif w == 'малооблачно':
    w = 10
elif w == 'сильный туман':
    w = 11
elif w == 'гроза':
    w = 12
elif w == 'пыль, ветер':
    w = 13
elif w == 'слабая морось':
    w = 14
elif w == 'снег':
    w = 15
elif w == 'ливневая крупа':
    w = 16
c = st.slider('Облачность (%)', 0, 100, 100)
Cl = st.selectbox('Тип облаков', ('Облаков нет', 'Слоисто-кучевые, образовавшиеся не из кучевых.',
                                  'Слоисто-кучевых, слоистых, кучевых или кучево-дождевых облаков нет.',
                                  'Кучево-дождевые лысые с кучевыми, слоисто-кучевыми или слоистыми, либо без них.',
                                  'Слоисто-кучевые, образовавшиеся из кучевых.',
                                  'Кучевые средние или мощные или вместе с кучевыми разорванными, или с кучевыми плоскими, или со слоисто-кучевыми, либо без них; основания всех этих облаков расположены на одном уровне.',
                                  'Кучевые плоские или кучевые разорванные, или те и другие вместе, не относящиеся к облакам плохой погоды.',
                                  'Кучево-дождевые волокнистые (часто с наковальней), либо с кучево-дождевыми лысыми, кучевыми, слоистыми, разорванно-дождевыми, либо без них.',
                                  'Слоистые туманообразные или слоистые разорванные, либо те и другие, но не относящиеся к облакам плохой погоды.'))
if Cl == 'Облаков нет':
    Cl = 0
elif Cl == 'Слоисто-кучевые, образовавшиеся не из кучевых.':
    Cl = 1
elif Cl == 'Слоисто-кучевых, слоистых, кучевых или кучево-дождевых облаков нет.':
    Cl = 2
elif Cl == 'Кучево-дождевые лысые с кучевыми, слоисто-кучевыми или слоистыми, либо без них.':
    Cl = 3
elif Cl == 'Слоисто-кучевые, образовавшиеся из кучевых.':
    Cl = 4
elif Cl == 'Кучевые средние или мощные или вместе с кучевыми разорванными, или с кучевыми плоскими, или со слоисто-кучевыми, либо без них; основания всех этих облаков расположены на одном уровне.':
    Cl = 5
elif Cl == 'Кучевые плоские или кучевые разорванные, или те и другие вместе, не относящиеся к облакам плохой погоды.':
    Cl = 6
elif Cl == 'Кучево-дождевые волокнистые (часто с наковальней), либо с кучево-дождевыми лысыми, кучевыми, слоистыми, разорванно-дождевыми, либо без них.':
    Cl = 7
elif Cl == 'Слоистые туманообразные или слоистые разорванные, либо те и другие, но не относящиеся к облакам плохой погоды.':
    Cl = 8
tS = st.number_input('Уровень снега (см)')
E = st.selectbox('Тип снега', ('Снега нет',
                               'Слежавшийся или мокрый снег (со льдом или без него), покрывающий менее половины поверхности почвы.',
                               'Ровный слой сухого рассыпчатого снега покрывает поверхность почвы полностью.',
                               'Слежавшийся или мокрый снег (со льдом или без него), покрывающий по крайней мере половину поверхности почвы, но почва не покрыта полностью.',
                               'Ровный слой слежавшегося или мокрого снега покрывает поверхность почвы полностью.',
                               'Сухой рассыпчатый снег покрывает по крайней мере половину поверхности почвы (но не полностью).',
                               'Сухой рассыпчатый снег покрывает меньше половины поверхности почвы.'))
if E == 'Снега нет':
    E = 0
elif E == 'Слежавшийся или мокрый снег (со льдом или без него), покрывающий менее половины поверхности почвы.':
    E = 1
elif E == 'Ровный слой сухого рассыпчатого снега покрывает поверхность почвы полностью.':
    E = 2
elif E == 'Слежавшийся или мокрый снег (со льдом или без него), покрывающий по крайней мере половину поверхности почвы, но почва не покрыта полностью.':
    E = 3
elif E == 'Ровный слой слежавшегося или мокрого снега покрывает поверхность почвы полностью.':
    E = 4
elif E == 'Сухой рассыпчатый снег покрывает по крайней мере половину поверхности почвы (но не полностью).':
    E = 5
elif E == 'Сухой рассыпчатый снег покрывает меньше половины поверхности почвы.':
    E = 6
P = st.number_input('Давление (мм)')
U = st.number_input('Влажность воздуха (%)')

if model == 'Жалагаш':
    X_new = np.array([[Time.hour, s, sH, T, p, w, c, Cl, tS, E, P, U]])
    dtest = xgb.DMatrix(X_new)
elif model == 'Шу':
    X_new = np.array([[Time.hour, s, sH, T, Cl, w, tS, E, P, U]])
    dtest = xgb.DMatrix(X_new)

y_pred = loaded_model.predict(dtest)
st.subheader('\n\n\n\nПрогнозируемое количество электроэнергии:')
if y_pred < 0 or s == 0:
    y_pred = 0
st.write(int(y_pred))

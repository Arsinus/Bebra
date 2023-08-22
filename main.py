import PySimpleGUI as sg
import time

# Задаем тему
sg.theme('DarkAmber')

# Создаем макет окна
layout = [[sg.Text('Пример использования нескольких шкал прогресса')],
          [sg.ProgressBar(100, orientation='h', size=(20, 20), key='progress1')],
          [sg.ProgressBar(100, orientation='h', size=(20, 20), key='progress2')],
          [sg.ProgressBar(100, orientation='h', size=(20, 20), key='progress3')]]

# Создаем окно
window = sg.Window('Пример шкал прогресса', layout)

# Имитируем обновление шкал прогресса
for i in range(100):
    event, values = window.read()
    window['progress1'].update_bar(i + 1)
    window['progress2'].update_bar(i + 1)
    window['progress3'].update_bar(i + 1)
    time.sleep(0.05)

window.close()

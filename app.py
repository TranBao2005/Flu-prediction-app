!pip install gradio openpyxl -q

import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

df = pd.read_csv('realistic_flu_dataset_1000.csv')

X = df[['Body_Temperature_C', 'Headache', 'Cough', 'Fatigue', 'Sore_Throat']]
y = df['Flu']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Perceptron(
    max_iter=1000,
    eta0=0.01,
    random_state=42
)

model.fit(X_train, y_train)

def convert_yes_no(value):
    return 1 if value == "Có" else 0

def predict_flu(temp, headache, cough, fatigue, sore_throat):

    headache = convert_yes_no(headache)
    cough = convert_yes_no(cough)
    fatigue = convert_yes_no(fatigue)
    sore_throat = convert_yes_no(sore_throat)

    input_data = [[temp, headache, cough, fatigue, sore_throat]]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        result = "Có khả năng bị cúm"
    else:
        result = "Không có dấu hiệu cúm"

    return f"""
Kết quả chẩn đoán:
{result}
"""

interface = gr.Interface(
    fn=predict_flu,

    inputs=[
        gr.Number(label="Nhiệt độ cơ thể (°C)"),

        gr.Radio(
            choices=["Có", "Không"],
            label="Đau đầu"
        ),

        gr.Radio(
            choices=["Có", "Không"],
            label="Ho"
        ),

        gr.Radio(
            choices=["Có", "Không"],
            label="Mệt mỏi"
        ),

        gr.Radio(
            choices=["Có", "Không"],
            label="Đau họng"
        )
    ],

    outputs=gr.Textbox(
        label="Kết quả",
        lines=5
    ),

    title="Hệ thống chẩn đoán bệnh cúm",
    description="Nhập triệu chứng để kiểm tra khả năng mắc cúm",

)

interface.launch()

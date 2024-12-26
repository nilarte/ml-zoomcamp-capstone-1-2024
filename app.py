import pickle
import gradio as gr

# Load the model and vectorizer
model_file = 'rf.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Define the prediction function
def predict_employee_performance(education, joiningyear, city, paymenttier, age, gender, everbenched, experienceincurrentdomain):
    # Create input dictionary
    employee_data = {
        "education": education,
        "joiningyear": joiningyear,
        "city": city,
        "paymenttier": paymenttier,
        "age": age,
        "gender": gender,
        "everbenched": everbenched,
        "experienceincurrentdomain": experienceincurrentdomain,
    }
    ['education', 'joiningyear', 'city', 'paymenttier', 'age', 'gender',
       'everbenched', 'experienceincurrentdomain', 'leaveornot']
    # Validate and preprocess inputs
    employee_data['joiningyear'] = int(employee_data['joiningyear'])
    employee_data['paymenttier'] = int(employee_data['paymenttier'])
    employee_data['age'] = int(employee_data['age'])
    employee_data['experienceincurrentdomain'] = int(employee_data['experienceincurrentdomain'])

    # Transform input and predict
    X = dv.transform([employee_data])
    y_pred = model.predict_proba(X)[0, 1]

    return {
        "Performance Probability": round(y_pred, 2),
    }

# Gradio interface
interface = gr.Interface(
    fn=predict_employee_performance,
    inputs=[
        gr.Dropdown(choices=["Bachelors", "Masters", "PhD"], label="Education"),
        gr.Number(label="Joining Year"),
        gr.Textbox(label="City"),
        gr.Number(label="Payment Tier (1, 2, 3)"),
        gr.Number(label="Age"),
        gr.Radio(choices=["Male", "Female"], label="Gender"),
        gr.Radio(choices=["Yes", "No"], label="Ever Benched (Yes/No)"),
        gr.Number(label="Experience in Current Domain (Years)"),
    ],
    outputs=[
        gr.Label(label="Prediction Results"),
    ],
    title="Employee Performance Prediction",
    description="Predict the likelihood of high performance based on employee data.",
)

# Launch the app
if __name__ == "__main__":
    interface.launch()

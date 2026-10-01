import joblib
import numpy as np

class StudentMarksPredictor():

    def __init__(self,model_path,scaler_path):

        self.model_path = model_path
        self.scaler_path = scaler_path

        self.model = None
        self.scaler = None

    def load_model(self):

        self.model = joblib.load(self.model_path)
        print("Model Load Successfully!")

    def load_scaler(self):
        self.scaler = joblib.load(self.scaler_path)
        print("Scaler load successfully!")

    def predict(self, study_hours, attendence, previous_marks):
        input_data = np.array([
            [study_hours,attendence,previous_marks]
        ])

        scaled_data = self.scaler.transform(input_data)
        
        prediction = self.model.predict(scaled_data)

        return prediction[0]

predictor = StudentMarksPredictor(
    r"C:\Users\Shubham\Desktop\students_marks_prediction\models\model.pkl",
    r"C:\Users\Shubham\Desktop\students_marks_prediction\models\scaler.pkl"
)


predictor.load_model()
predictor.load_scaler()

# student_hours = float(input("Enter study hours here: "))
# attendence = float(input("Enter student attendence here: "))
# previous_marks = float(input("Enter previous marks here: "))

# result = predictor.predict(
#     student_hours,
#     attendence,
#     previous_marks
# )


# print("Predicted Final Marks:", round(result,2))
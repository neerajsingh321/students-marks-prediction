from fastapi import FastAPI , HTTPException
from pydantic import BaseModel, Field
from src.predict import StudentMarksPredictor

class studentData(BaseModel):
    study_hours: float = Field(ge=0)
    attendance: float = Field(ge=0,le=100)
    previous_marks: float = Field(ge=0,le=100)


app = FastAPI()

@app.post("/predict")
def predict_marks(data:studentData):

    try:

        prediction = predictor.predict(
            data.study_hours,
            data.attendance,
            data.previous_marks
        )

        return {
           "study_hours":data.study_hours,
           "attendance":data.attendance,
           "previous_marks":data.previous_marks,
           "predicted_final_marks": round(float(prediction),2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


@app.get("/health")
def health_check():
    return {
        "status":"healthy",
        "model":"Student Marks Prediction"
    }


@app.get("/")
def home():
    return {"message":"Student Marks Prediction API is running"}

predictor = StudentMarksPredictor(
    r"C:\Users\Shubham\Desktop\students_marks_prediction\models\model.pkl",
    r"C:\Users\Shubham\Desktop\students_marks_prediction\models\scaler.pkl"
)

predictor.load_model()
predictor.load_scaler()


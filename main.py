from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import pickle

# Enable session support
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="some_secret_key")

templates = Jinja2Templates(directory="templates")

# Load the models and their parameters
models = {}
model_info = {}

# Load the Logistic Regression model
with open("./models/lr_pipeline.pkl", "rb") as file:
    lr_pipeline = pickle.load(file)
models["lr"] = lr_pipeline["pipeline"]
model_info["lr"] = {
    "parameters": lr_pipeline["parameters"],
    "f2_score": lr_pipeline["f2_score"],
    "confusion_matrix": "<br>".join(
        " - ".join(str(cell) for cell in row) for row in lr_pipeline["confusion_matrix"]
    ),
    "classification_report": lr_pipeline["classification_report"],
}

# Load the Random Forest model
with open("./models/rf_pipeline.pkl", "rb") as file:
    rf_pipeline = pickle.load(file)
models["rf"] = rf_pipeline["pipeline"]
model_info["rf"] = {
    "parameters": rf_pipeline["parameters"],
    "f2_score": rf_pipeline["f2_score"],
    "confusion_matrix": rf_pipeline["confusion_matrix"],
    "classification_report": rf_pipeline["classification_report"],
}

# Load the SVM model
with open("./models/svm_pipeline.pkl", "rb") as file:
    svm_pipeline = pickle.load(file)
models["svm"] = svm_pipeline["pipeline"]
model_info["svm"] = {
    "parameters": svm_pipeline["parameters"],
    "f2_score": svm_pipeline["f2_score"],
    "confusion_matrix": svm_pipeline["confusion_matrix"],
    "classification_report": svm_pipeline["classification_report"],
}

# Load the SGD model
with open("./models/sgd_pipeline.pkl", "rb") as file:
    sgd_pipeline = pickle.load(file)
models["sgd"] = sgd_pipeline["pipeline"]
model_info["sgd"] = {
    "parameters": sgd_pipeline["parameters"],
    "f2_score": sgd_pipeline["f2_score"],
    "confusion_matrix": sgd_pipeline["confusion_matrix"],
    "classification_report": sgd_pipeline["classification_report"],
}

# Load the baseline model
with open("./models/baseline_rf_model/trained_rf_model.pkl", "rb") as file:
    baseline_model = pickle.load(file)

# Load the saved vectorizer model for baseline model
with open("./models/baseline_rf_model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/{model_name}")
async def predict_api(model_name: str, message: str = Form(...)):
    if model_name in models:
        model = models[model_name]
        prediction = model.predict([message])[0]
        return {"prediction": int(prediction)}
    else:
        return JSONResponse(status_code=404, content={"error": "Model not found"})


@app.get("/user/baseline", response_class=HTMLResponse)
async def predict_baseline(request: Request):
    past_descriptions = request.session.get("past_descriptions", [])
    return templates.TemplateResponse(
        "baseline.html", {"request": request, "past_descriptions": past_descriptions}
    )


@app.get("/user/{model_name}", response_class=HTMLResponse)
async def predict_user(request: Request, model_name: str):
    if model_name in models:
        model = models[model_name]
        model_info_data = model_info[model_name]
        past_descriptions = request.session.get("past_descriptions", [])
        return templates.TemplateResponse(
            "model.html",
            {
                "request": request,
                "model_name": model_name,
                "model_info": model_info_data,
                "past_descriptions": past_descriptions,
            },
        )
    else:
        return templates.TemplateResponse(
            "error.html", {"request": request, "error_message": "Model not found"}
        )


@app.post("/user/{model_name}", response_class=HTMLResponse)
async def predict_user_post(
    request: Request, model_name: str, description: str = Form(...)
):
    if model_name in models:
        model = models[model_name]
        model_info_data = model_info[model_name]
        past_descriptions = request.session.get("past_descriptions", [])
        prediction = model.predict([description])[0]
        prediction_value = int(prediction)
        answer = "Looks legit"
        if prediction_value:
            answer = "Seems fraudulent"
        past_descriptions.insert(
            0, {"description": description, "answer": answer, "model": model_name}
        )
        request.session["past_descriptions"] = past_descriptions
        return RedirectResponse(f"/user/{model_name}", status_code=303)
    elif model_name == "baseline":
        past_descriptions = request.session.get("past_descriptions", [])
        # Transform the input text data using the loaded vectorizer
        description_transformed = tfidf_vectorizer.transform([description])

        # Make a prediction using the baseline_model
        prediction = baseline_model.predict(description_transformed)[0]
        # Convert prediction to a regular Python integer
        prediction_value = int(prediction)
        answer = "Looks legit"
        if prediction_value:
            answer = "Seems fraudulent"
        past_descriptions.insert(
            0, {"description": description, "answer": answer, "model": model_name}
        )
        request.session["past_descriptions"] = past_descriptions
        return RedirectResponse("/user/baseline", status_code=303)
    else:
        return templates.TemplateResponse(
            "error.html", {"request": request, "error_message": "Model not found"}
        )


@app.get("/clear")
async def clear(request: Request):
    past_descriptions = request.session.get("past_descriptions", [])
    past_descriptions.clear()
    request.session["past_descriptions"] = past_descriptions
    return RedirectResponse("/", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

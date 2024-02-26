"""from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.post("/predict/")
async def predict(description: str = Form(...)):
    words = description.split()
    if not words:
        return {"error": "Description cannot be empty"}
    return {"result": words[0]}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import pickle
import json


# Enable session support
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="some_secret_key")

templates = Jinja2Templates(directory="templates")

# Load the model
with open("trained_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the saved vectorizer model
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    past_descriptions = request.session.get("past_descriptions", [])
    return templates.TemplateResponse(
        "index.html", {"request": request, "past_descriptions": past_descriptions}
    )


@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, description: str = Form(...)):
    past_descriptions = request.session.get("past_descriptions", [])
    # Transform the input text data using the loaded vectorizer
    description_transformed = tfidf_vectorizer.transform([description])

    # Make a prediction using the loaded model
    prediction = model.predict(description_transformed)[0]
    # Convert prediction to a regular Python integer
    prediction_value = int(prediction)

    answer = "This looks like a legit description."
    if prediction_value:
        answer = "Fraudulent description detected !!!"

    past_descriptions.insert(0, {"description": description, "answer": answer})
    # Convert non-serializable elements to JSON strings
    request.session["past_descriptions"] = past_descriptions

    return RedirectResponse("/", status_code=303)


@app.get("/clear")
async def clear(request: Request):
    # Retrieve past_descriptions from session
    past_descriptions = request.session.get("past_descriptions", [])

    # Clear the past_descriptions list
    past_descriptions.clear()

    # Update the session data
    request.session["past_descriptions"] = past_descriptions

    return RedirectResponse("/", status_code=303)

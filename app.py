import h2o
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

app = FastAPI()
h2o.init()

# Ensure you are using the correct path to your model file
MODEL_PATH = r"C:\Users\PAVSHAN\Downloads\Pred\StackedEnsemble_AllModels_3_AutoML_1_20250902_132009"
model = h2o.load_model(MODEL_PATH)

templates = Jinja2Templates(directory=".")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    # Pass default None values for the prediction variables on initial page load
    return templates.TemplateResponse("form.html", {
        "request": request,
        "earned_hrs_pred": None,
        "perf_pct_pred": None
    })

@app.post("/", response_class=HTMLResponse)
def predict(
    request: Request,
    future_date: str = Form(...),
    elaps_acty_time: float = Form(...),
    lbr_std_time: float = Form(...)
):
    input_dict = {
        "EMP_ID": "119915",
        "LBR_WORK_DATE": future_date,
        "LBR_CHRG_NO": "BQNEXT",
        "ACTU_START_TS": f"{future_date} 08:00",
        "ACTU_END_TS": f"{future_date} 12:00",
        "SHIFT_CD": 3,  # Keep as a number
        "ELAPS_ACTY_TIME": elaps_acty_time,
        "QTY_CMPLTD": "10",
        "WO_OPR_NO": "020",
        "WC_POOL": "88",
        "WC_STN": "J16W",
        "ITEM_NO": "18-44737-002",
        "LBR_STD_TIME": lbr_std_time
    }
    
    feature_cols = list(input_dict.keys())
    
    # --- FIX IS HERE ---
    # Remove "SHIFT_CD" from this list because the model expects it to be a number
    categorical_cols = ["EMP_ID", "LBR_CHRG_NO", "QTY_CMPLTD", "WO_OPR_NO", "WC_POOL", "WC_STN", "ITEM_NO"]
    # --- END OF FIX ---
    
    row = pd.DataFrame([input_dict])[feature_cols]

    for col in categorical_cols:
        row[col] = row[col].astype(str)

    column_types = {col: 'enum' for col in categorical_cols}
    pred_h2o = h2o.H2OFrame(row, column_types=column_types)

    pred_result = model.predict(pred_h2o).as_data_frame().iloc[0]
    
    earned_hrs_pred = pred_result.get("predict", 0)
    perf_pct_pred = pred_result.get("p1") # This might be None if not applicable

    return templates.TemplateResponse("form.html", {
        "request": request,
        "earned_hrs_pred": earned_hrs_pred,
        "perf_pct_pred": perf_pct_pred,
        "future_date": future_date,
        "elaps_acty_time": elaps_acty_time,
        "lbr_std_time": lbr_std_time,
    })
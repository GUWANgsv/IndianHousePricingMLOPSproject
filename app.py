from flask import Flask, render_template, request, jsonify
import os
import sys
import traceback
import pandas as pd

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

app = Flask(__name__)


def _load_predict_pipeline():
	from src.IndianHousePricingMLOPSproject.pipelines.Prediction_Pipeline import PredictPipeline
	return PredictPipeline()


def _as_float(value, default=0.0):
	try:
		if value in (None, ""):
			return default
		return float(value)
	except Exception:
		return default


def _as_int(value, default=0):
	try:
		if value in (None, ""):
			return default
		return int(float(value))
	except Exception:
		return default


def _build_input_frame(payload):
	"""Build a one-row DataFrame that matches the raw training schema loosely."""
	row = {
		"carpetArea": _as_float(payload.get("carpetArea")),
		"carpetAreaUnit": payload.get("carpetAreaUnit", "Sq-ft"),
		"propertyType": payload.get("propertyType", "Multistorey Apartment"),
		"postedOn": payload.get("postedOn", "Jun 20, '23"),
		"noOfLifts": _as_float(payload.get("noOfLifts")),
		"maintenanceChargesFrequency": payload.get("maintenanceChargesFrequency", "9"),
		"maintenanceCharges": _as_float(payload.get("maintenanceCharges")),
		"locality": payload.get("locality", "9"),
		"furnishing": payload.get("furnishing", "Semi-Furnished"),
		"flrNum": payload.get("flrNum", "1"),
		"firstMonthCharges": _as_float(payload.get("firstMonthCharges")),
		"facing": payload.get("facing", "9"),
		"totalFlrNum": _as_float(payload.get("totalFlrNum")),
		"city": payload.get("city", "Patna"),
		"brokerage": _as_float(payload.get("brokerage")),
		"bedrooms": _as_int(payload.get("bedrooms"), 1),
		"bathrooms": _as_int(payload.get("bathrooms"), 1),
		"balconies": _as_int(payload.get("balconies"), 0),
		"securityDeposit": _as_float(payload.get("securityDeposit")),
		"RentOrSale": payload.get("RentOrSale", "Rent"),
		"Water_Storage": _as_int(payload.get("Water_Storage"), 0),
		"Waste_Disposal": _as_int(payload.get("Waste_Disposal"), 0),
		"Visitor_Parking": _as_int(payload.get("Visitor_Parking"), 0),
		"Vaastu_Compliant": _as_int(payload.get("Vaastu_Compliant"), 0),
		"Swimming_Pool": _as_int(payload.get("Swimming_Pool"), 0),
		"Security": _as_int(payload.get("Security"), 0),
		"Reserved_Parking": _as_int(payload.get("Reserved_Parking"), 0),
		"Power_Back_Up": _as_int(payload.get("Power_Back_Up"), 0),
		"Park": _as_int(payload.get("Park"), 0),
		"Gymnasium": _as_int(payload.get("Gymnasium"), 0),
	}
	# Keep the raw schema stable for the model pipeline; any missing fields get defaults here.
	for key in ["securityDeposit", "noOfLifts", "maintenanceCharges", "firstMonthCharges", "brokerage"]:
		row[key] = _as_float(payload.get(key), row.get(key, 0.0))
	return pd.DataFrame([row])


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
	payload = request.get_json(silent=True)
	if payload is None:
		payload = request.form.to_dict()

	try:
		features = _build_input_frame(payload)
		pipeline = _load_predict_pipeline()
		preds = pipeline.predict(features)
		return jsonify({"prediction": float(preds[0])})
	except Exception as e:
		traceback.print_exc()
		return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True, load_dotenv=False)

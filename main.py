import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib  # ή pickle αν χρησιμοποιείτε αυτό για το μοντέλο σας

app = Flask(__name__)

# Εδώ θα φορτώσετε το εκπαιδευμένο μοντέλο σας
# Αν το μοντέλο παράγεται από το train_evaluate_monthly.py,
# βεβαιωθείτε ότι το έχετε κάνει save ως .pkl
MODEL_PATH = "model.pkl"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Λήψη δεδομένων από το αίτημα (request)
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # Φόρτωση μοντέλου (καλό είναι να γίνεται εκτός της predict 
        # για ταχύτητα, αλλά για αρχή το βάζουμε εδώ)
        model = joblib.load(MODEL_PATH)
        
        # Πραγματοποίηση πρόβλεψης
        prediction = model.predict(df)
        
        return jsonify({
            "status": "success",
            "prediction": prediction.tolist()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # ΠΡΟΣΟΧΗ: Το Cloud Run απαιτεί τη θύρα από τη μεταβλητή περιβάλλοντος PORT
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

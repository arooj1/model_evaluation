'''
FLASK APP 

@purpose: Model Confidence Level 
@author: Arooj Ahmed Qureshi 
@contributors: Habib and Eugene
@company: EnPowered INC
'''

import os
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

import Code.model_confidence_level as mc


#################################################
# Flask Setup
#################################################
app = Flask(__name__)

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/nyiso")
def nyiso():
    path = 'Data/NYISO_September_combine_data.csv'
    #path = '../Data/NYISO_August_combine_data.csv'
    model_region = 'NYISO'
    thres = 29000
    month = 'September, 2021'
    peak_day = mc.model_confidence(path, 'NYISO', thres )

    #path = '../Data/ERCOT_September_combine_data.csv'
    #peak_day = model_confidence(path, 'ERCOT', 70000)
    day_result_base, day_result_ensemble, hourly_result_base= peak_day()
    return render_template("nyiso.html", thres = thres, month = month, base_model = day_result_base, ensemble_model = day_result_ensemble, hour_accuracy = hourly_result_base)






if __name__ == "__main__":
    app.run()

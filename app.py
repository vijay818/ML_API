import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

custbehav_model = pickle.load(open('custbehav_model.pkl','rb'))
churnpred_model = pickle.load(open('churnpred_model.pkl','rb'))
tsforecast_model = pickle.load(open('ts_sales_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result',methods=['POST'])
def result():
    if 'mba_button' in request.form:
        return render_template('mba_form.html')
    elif 'FrequentPurchases' in request.form:
        df = pd.read_csv("frequent.csv")
        features = request.form.get('FrequentPurchases')
        #int_features = [int(x) for x in request.form.values()]
        #final_features = [np.array(int_features)]
        #prediction = model.predict(final_features)
        output = df[df.antecedents == features]["consequents"][:1]
        s = output.to_string(index = False)
        return render_template('mba_form.html', prediction_text=s)
    elif 'custbehav_button' in request.form:
        return render_template('custbehav_form.html')
    elif 'PageValues' in request.form:
        #int_fetss = [x for x in request.form.values()]
        #return str(int_fetss)
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = custbehav_model.predict(final_features)
        output = prediction[0]
        return render_template('custbehav_form.html', prediction_text = 'Customer Propensity is {}'.format(output))
    elif 'custchurn_button' in request.form:
        #return "Custchurn button clicked"
        return render_template('churnpred_form.html')   
    elif 'CurrentEquipmentDays' in request.form:
        #for x in request.form.values():
        #int_fets = [x for x in request.form.values() if x!= 'Submit']
        #return str(int_fets)
        int_features = [float(x) for x in request.form.values() if x!= 'Submit']
        final_features = [np.array(int_features)]
        prediction = churnpred_model.predict(final_features)
        output = prediction[0]
        return render_template('churnpred_form.html', prediction_text = 'Churn value is {}'.format(output))
    elif 'tsforecast_button' in request.form:
        #return "Timeseries Forecast button clicked"
        return render_template('tsforecast_form.html')
    elif 'NumberofMonths' in request.form:
        #return "Timeseries Forecast response button clicked"
        prediction = tsforecast_model.predict(start='2019-07-01', to='2019-09-01',dynamic=True)
        output = prediction.to_string(index=False)
        lst = []        
        for i in prediction:
            i = int(i)
            lst.append(i)
        return render_template('tsforecast_form.html', prediction_text = 'Sales Forecast for next 3 months with an RMSE of 101.3 is: {}'.format(lst))
        

if __name__ == "__main__":
    app.run(debug=True)
    



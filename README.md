Project Overview
In the die-casting industry, developing steel with optimal mechanical properties has always been a challenging task. Traditionally, this process involved trial-and-error methods — a time-consuming and expensive approach. Our team at a die-casting equipment manufacturing company faced this challenge while assisting a customer in improving their steel grade for enhanced fatigue strength.

To address this, I developed a Machine Learning (ML) solution that predicts Fatigue Strength based on steel's chemical composition and heat treatment parameters. This app helps engineers optimize steel grades efficiently, reducing both material waste and production time.

Problem Statement
Before developing this solution, our team relied on a trial-and-error method to test different steel compositions — consuming weeks or even months. Each failed attempt wasted valuable resources and increased costs.

The goal was to:
✅ Predict Fatigue Strength and other key mechanical properties without physical trials.
✅ Provide actionable insights to engineers through SHAP Analysis and What-If Analysis.
✅ Develop a user-friendly interface for offline use in industrial settings.
✅ Deploy an online version for demonstration and learning purposes.

 Features
✅ Predicts Fatigue Strength using ML model.
✅ Estimates additional mechanical properties:

Ultimate Tensile Strength (UTS)
Yield Strength (YS)
Brinell Hardness (HB)
Vickers Hardness (HV)
Rockwell Hardness (HRB & HRC)
Ductility (%)
Toughness (MPa√m)
✅ Provides SHAP Analysis to explain feature importance.
✅ Offers What-If Analysis to visualize property changes with varying parameters.
✅ Generates downloadable Steel Recipe Reports in CSV and PDF formats.
✅ Includes preset values for Low, Medium, and High Carbon Steel compositions.
✅ Designed for both offline use (via desktop app) and online use (via Streamlit Cloud).
Installation & Setup (Offline App)
Step 1: Clone the Repository
bash
Copy code
git clone https://github.com/msmukul51/Steel-strength-prediction.git
cd Steel-strength-prediction
Step 2: Create a Virtual Environment
bash
Copy code
python -m venv venv
Step 3: Activate the Virtual Environment
For Windows:

bash
Copy code
venv\Scripts\activate
For Mac/Linux:

bash
Copy code
source venv/bin/activate
Step 4: Install Dependencies
bash
Copy code
pip install -r requirements.txt
Step 5: Run the App
For Online Version (Streamlit):

bash
Copy code
streamlit run app.py
For Offline Desktop App (Tkinter):

bash
Copy code
python offline_app.py
 Creating an .exe File for Desktop Use
To package the offline app into a standalone executable:

bash
Copy code
pyinstaller --onefile --noconsole --icon=icon.ico offline_app.py
The .exe file will be located in the /dist folder.
Copy pipe.pkl and other required files into the same folder.
 Access the Online App
 Steel Fatigue Strength Prediction App

How to Use
Enter steel parameters such as:
Normalizing Temperature
Carbon Content (%)
Hardening Time
And other chemical compositions or heat treatment details.
Click "Predict Steel Strength" to view the predicted values.
Use "What-If Analysis" to explore how parameter changes impact fatigue strength.
Download detailed steel recipes and property reports in CSV or PDF format.
Challenges Faced
 Data Collection: Finding sufficient data was challenging. I collected data from material testing labs and customer samples, resulting in a smaller dataset than ideal.
 Model Training: Since the data was non-linear, I tested multiple algorithms like Random Forest, XGBoost, and LightGBM before selecting the best-performing model.
 Fine-tuning: Extensive hyperparameter tuning was done to improve model accuracy.
 Feature Expansion: To meet client needs, I added features like:

SHAP Analysis for transparency.
What-If Analysis for interactive insights.
 Deployment: While the app was designed for offline use in industrial setups, I deployed it on Streamlit Cloud for my own learning and demonstration purposes.
 Impact
 Reduced trial-and-error testing in steel grade development.
 Saved significant time and cost by predicting fatigue strength directly.
 Provided valuable insights into key factors affecting steel performance through SHAP Analysis.
 Enabled engineers to explore custom scenarios via What-If Analysis, improving decision-making.

 Learning Outcomes
 Improved skills in data collection strategies for small datasets.
 Learned to test and compare multiple algorithms for non-linear data.
 Developed expertise in SHAP Analysis and feature impact visualization.
 Mastered designing both online (Streamlit) and offline (Tkinter) apps.
 Understood the importance of setting realistic parameter boundaries by identifying min and max values in the dataset.

 Connect with Me
If you’d like to discuss this project, feel free to connect on LinkedIn or drop a message.

 "Turning data into insights and engineering solutions that matter."

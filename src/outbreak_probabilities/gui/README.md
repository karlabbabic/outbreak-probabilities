# PMO GUI
This project is a Python Tkinter-based graphical user interface that enables users to input three incidence data points from an outbreak and obtain a prediction indicating whether the outbreak is likely to go extinct or escalate into a mass outbreak.

The interface is designed for simplicity and clarity, making the tool accessible to both technical and non-technical users including Hospital staff who report incidences, epidemiologists, public health personnel, and researchers.

# Core Features of the Simple and intuitive GUI will include:

- Input 1 - boxes for three incidence values
- Input 2 - fields for three incidence values, Built-in validation to prevent invalid entries (numeric-only handling + error prompts).
- Input 3 - Button interface for running predictions and resetting inputs

- Output 1 - Prediction engine estimating: Probability of extinction, Probability of outbreak
- Output 2 - Display/Visualization of prediction results (Graphs), 
- Output 3 - Export predictions (CSV, PDF, PNG)
- Output 4 - Cross-platform compatibility (Windows/Mac/Linux)

Button interface for running predictions and resetting inputs
User Guidelines
Report output to Public Health and Other regulatory Agencies Automatically
# GUI OVERVIEW:
- Main Window (Homepage.py) - Application title, A brief description label, Navigation to the input section (or displayed on the same window depending on layout)
- INPUT PAGE (inputframe.py): A Tkinter Frame containing - Label + Entry field for incidences (1,2,3), Input validation using - register(), validation inside the call box, Reset Inputs (clears Entry widgets), Run Prediction (calls the model), 
- PREDICTION FRAME (predictionframe.py):After clicking Run Prediction, the system displays - Extinction probability, Outbreak probability - MODELS***
- RESULT FRAME (Resultframe.py) - After runing the prediction, the result is visualised with colour coded output (Green for extenction and Red for PMO - Matplotlib window pop-up for visual charts), Interpretation text (e.g., “High risk of outbreak”), Export as PDFor PNG, Exit Application.
- CONTROL BOTTONS - Run Prediction (calls the model), Reset Inputs (clears Entry widgets), Exit Application
# Other Essentials (Optional)
- pop-up labels or small description text below entries
- Provision of input boxes for more incidences beyond 3
- Interoperability to communicate with other useful incidences GUI for public health outbreaks



 




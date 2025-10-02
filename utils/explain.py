import shap
import matplotlib.pyplot as plt

def explain_prediction(model, scaled_input, feature_names):
    """ Return SHAP explanation figure """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, scaled_input, feature_names=feature_names, plot_type="bar", show=False)
    return fig

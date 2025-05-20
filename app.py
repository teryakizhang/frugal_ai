import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Frugal Model Selector", layout="wide")

st.title("🔋 Frugal AI Model Rating Tool")

# Description with formula
st.markdown(
    r"""
    **Frugal Rating Formula**:
    
    $$
    \mathrm{frugal}_i = \alpha \frac{F_i - F_{\min}}{F_{\max} - F_{\min}} 
    + \beta \Bigl(1 - \frac{T_i - T_{\min}}{T_{\max} - T_{\min}}\Bigr)
    + \gamma \Bigl(1 - \frac{I_i - I_{\min}}{I_{\max} - I_{\min}}\Bigr)
    $$
    
    where:
    - $F_i$ = F1 score of model $i$
    - $T_i$ = training energy (kWh)
    - $I_i$ = inference energy (kWh per 1K inferences)
    - $\alpha,\beta,\gamma$ = weights that sum to 1

    The above formula first normalizes the F1 score and energy consumptions. The training and inference energy scores are then inverted so that lower energy consumption results in a higher score. 
    The final frugal rating is a weighted sum of these normalized values.
    """,
    unsafe_allow_html=True
)

# Sidebar for config
st.sidebar.header("Configuration")
st.sidebar.markdown(
    r"""
    **Note**: The weights are automatically normalized to sum to 1.
    """
)
# Sliders for weights 
alpha_input = st.sidebar.slider("F1 weight", 0.0, 1.0, 0.33, 0.01)
beta_input = st.sidebar.slider("Training energy weight", 0.0, 1.0, 0.33, 0.01)
gamma_input = st.sidebar.slider("Inference energy weight", 0.0, 1.0, 0.34, 0.01)

use_grid_factor = st.sidebar.checkbox(
    "Use grid emission factor for energy consumption",
    value=True,
    help="If unchecked, the energy consumption will be used to calculate the rating."
)
grid_factor = st.sidebar.number_input(
    "Grid emission factor (gCO2e per kWh)",
    min_value=0.0,
    value=175.0,
    step=0.01,
    format="%.2f",
    label_visibility="collapsed",
    disabled=not use_grid_factor,
)


# Result upload
st.header("Upload your model results CSV")
uploaded_file = st.file_uploader(
    "CSV with columns: model_name, f1_score, training_energy_kwh, inference_energy_kwh",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ["model_name", "f1_score", "training_energy_kwh", "inference_energy_kwh"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must include columns: {required_cols}")
        st.stop()

    # F1 floor filter
    F_min_all, F_max_all = df['f1_score'].min(), df['f1_score'].max()
    f1_floor = st.sidebar.slider(
        "F1 minimum threshold", float(F_min_all), float(F_max_all), float(F_min_all), 0.01
    )
    df = df[df['f1_score'] >= f1_floor]
    if df.empty:
        st.error("No models meet the F1 floor threshold.")
        st.stop()

    # Normalize weights
    total = alpha_input + beta_input + gamma_input
    if total == 0:
        st.sidebar.error("At least one weight must be > 0.")
        st.stop()
    alpha = alpha_input / total
    beta = beta_input / total
    gamma = gamma_input / total
    st.sidebar.markdown(f"**Normalized weights:**  α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

    if use_grid_factor:
        df['train_emission'] = df['training_energy_kwh'] * grid_factor
        df['inf_emission']   = df['inference_energy_kwh'] * grid_factor
    else:
        df['train_emission'] = df['training_energy_kwh']
        df['inf_emission']   = df['inference_energy_kwh']

    st.write("## Filtered Data")
    st.dataframe(df)

    # Compute mins and maxes on filtered set
    F_min, F_max = df['f1_score'].min(), df['f1_score'].max()
    T_min, T_max = df['train_emission'].min(), df['train_emission'].max()
    I_min, I_max = df['inf_emission'].min(), df['inf_emission'].max()

    # Normalize and invert energies
    df['f1_norm'] = (df['f1_score'] - F_min) / (F_max - F_min)
    df['train_norm'] = (df['train_emission'] - T_min) / (T_max - T_min)
    df['inf_norm'] = (df['inf_emission'] - I_min) / (I_max - I_min)
    df['train_inv'] = 1 - df['train_norm']
    df['inf_inv'] = 1 - df['inf_norm']

    # Compute weighted frugal rating
    df['frugal_rating'] = (
        alpha * df['f1_norm'] + beta * df['train_inv'] + gamma * df['inf_inv']
    )

    # Final df
    df_sorted = df.sort_values('frugal_rating', ascending=False).reset_index(drop=True)

    st.write("## Frugal Rating Results")
    st.dataframe(df_sorted[['model_name', 'frugal_rating']])

    # Bar chart
    st.write("## Frugal Rating Bar Chart")
    bar_chart = alt.Chart(df_sorted).mark_bar().encode(x=alt.X('model_name', sort=None), y='frugal_rating')
    st.altair_chart(bar_chart, use_container_width=True)
else:
    st.info("Upload a CSV file to get started.")

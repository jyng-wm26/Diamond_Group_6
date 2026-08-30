"""Interactive BMDS2003 diamond price prediction prototype."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from diamond_ml import (
    MODEL_FEATURES,
    MODEL_RATIONALE,
    RANDOM_STATE,
    calculate_permutation_importance,
    evaluate_models,
    train_model_bundle,
)
from diamond_3d import create_diamond_figure
st.set_page_config(
    page_title="Diamond intelligence studio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
DATA_FILE_CANDIDATES = [
    APP_DIR / "Diamonds Prices2022.csv",
    APP_DIR / "Diamonds Prices2022(1).csv",
]
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOUR_ORDER = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
NUMERICAL_COLUMNS = ["carat", "depth", "table", "price", "x", "y", "z"]
PAGE_OPTIONS = [
    "Project overview",
    "Explore data",
    "Report figures",
    "Model comparison",
    "Model diagnostics",
    "Price predictor",
]

REPORT_FIGURES = {
    "Section 1 - Business understanding": [
        (
            "Figure 1.1",
            "Project workflow aligned with CRISP-DM",
            "section1_business_understanding/assets/figure_1_1_crisp_dm_project_workflow.png",
            "Project overview > CRISP-DM workflow",
        ),
    ],
    "Section 2 - Data understanding": [
        ("Figure 2.1", "Distribution of observed diamond prices", "section2_data_understanding/assets/figure_2_1_price_distribution.png", "Explore data > Distribution"),
        ("Figure 2.2", "Frequency distribution of diamond carat weight", "section2_data_understanding/assets/figure_2_2_carat_distribution.png", "Report figures (exact report output)"),
        ("Figure 2.3", "Carat weight versus diamond price", "section2_data_understanding/assets/figure_2_3_carat_vs_price.png", "Explore data > Carat relationship"),
        ("Figure 2.4", "Price by cut quality", "section2_data_understanding/assets/figure_2_4_price_by_cut.png", "Explore data > Category comparison > cut"),
        ("Figure 2.5", "Price by colour grade", "section2_data_understanding/assets/figure_2_5_price_by_color.png", "Explore data > Category comparison > color"),
        ("Figure 2.6", "Price by clarity grade", "section2_data_understanding/assets/figure_2_6_price_by_clarity.png", "Explore data > Category comparison > clarity"),
        ("Figure 2.7", "Correlation heatmap of numerical features", "section2_data_understanding/assets/figure_2_7_correlation_heatmap.png", "Explore data > Correlation"),
        ("Figure 2.8", "Physical dimensions versus diamond price", "section2_data_understanding/assets/figure_2_8_dimensions_vs_price.png", "Report figures (exact report output)"),
        ("Figure 2.9", "Price distribution across carat groups", "section2_data_understanding/assets/figure_2_9_price_by_carat_group.png", "Report figures (exact report output)"),
        ("Figure 2.10", "Depth percentage versus diamond price", "section2_data_understanding/assets/figure_2_10_depth_vs_price.png", "Report figures (exact report output)"),
        ("Figure 2.11", "Table percentage versus diamond price", "section2_data_understanding/assets/figure_2_11_table_vs_price.png", "Report figures (exact report output)"),
        ("Figure 2.12", "Median price by cut and colour", "section2_data_understanding/assets/figure_2_12_cut_color_heatmap.png", "Report figures (exact report output)"),
        ("Figure 2.13", "Median price by cut and clarity", "section2_data_understanding/assets/figure_2_13_cut_clarity_heatmap.png", "Report figures (exact report output)"),
        ("Figure 2.14", "Price per carat across quality grades", "section2_data_understanding/assets/figure_2_14_price_per_carat_quality.png", "Report figures (exact report output)"),
        ("Figure 2.15", "Profile of the top-10% price segment", "section2_data_understanding/assets/figure_2_15_top_price_segment.png", "Report figures (exact report output)"),
    ],
    "Section 3 - Data preparation": [
        ("Figure 3.1", "Missing-value inspection", "section3_data_preparation/assets/figure_3_1_missing_values.png", "Project overview > Preparation evidence"),
        ("Figure 3.2", "Cleaning implementation", "section3_data_preparation/assets/figure_3_2_cleaning_implementation.png", "Project overview > Preparation evidence"),
        ("Figure 3.3", "Records removed during cleaning", "section3_data_preparation/assets/figure_3_3_cleaning_impact.png", "Project overview > Preparation evidence"),
        ("Figure 3.4", "Numerical boxplots before outlier treatment", "section3_data_preparation/assets/figure_3_4_iqr_boxplots.png", "Report figures (exact report output)"),
        ("Figure 3.5", "IQR outlier flags by numerical variable", "section3_data_preparation/assets/figure_3_5_iqr_counts.png", "Report figures (exact report output)"),
        ("Figure 3.6", "Dimension-consistency review", "section3_data_preparation/assets/figure_3_6_dimension_consistency.png", "Project overview > Preparation evidence"),
        ("Figure 3.7", "Outlier detection and treatment implementation", "section3_data_preparation/assets/figure_3_7_outlier_treatment_code.png", "Project overview > Preparation evidence"),
        ("Figure 3.8", "Preprocessing configuration", "section3_data_preparation/assets/figure_3_4_preprocessing_configuration.png", "Project overview > Model design"),
        ("Figure 3.9", "Reproducible 80:20 train-test split", "section3_data_preparation/assets/figure_3_9_train_test_split.png", "Project overview > Preparation evidence"),
    ],
    "Section 4 - Modelling": [
        ("Figure 4.1", "Model configuration and training implementation", "section4_modelling/assets/figure_4_1_model_training_code.png", "Project overview > Model design"),
        ("Figure 4.2", "MAE and RMSE comparison", "section4_modelling/assets/figure_4_2_test_error_comparison.png", "Model comparison > Metric comparison"),
        ("Figure 4.3", "Actual versus predicted prices for all models", "section4_modelling/assets/figure_4_3_actual_vs_predicted.png", "Model comparison > Test fit and Model diagnostics > Fit"),
        ("Figure 4.4", "XGBoost validation-only early stopping", "section4_modelling/assets/figure_4_4_xgboost_early_stopping.png", "Model comparison > XGBoost tuning"),
        ("Figure 4.5", "XGBoost permutation feature importance", "section4_modelling/assets/figure_4_5_xgboost_permutation_importance.png", "Model diagnostics > Feature importance"),
    ],
    "Section 5 - Evaluation": [
        ("Figure 5.1", "RMSE comparison for all models", "section5_evaluation/assets/figure_5_1_rmse_comparison.png", "Model comparison > Metric comparison > RMSE"),
        ("Figure 5.2", "MAE comparison for all models", "section5_evaluation/assets/figure_5_2_mae_comparison.png", "Model comparison > Metric comparison > MAE"),
        ("Figure 5.3", "R2 comparison for all models", "section5_evaluation/assets/figure_5_3_r2_comparison.png", "Model comparison > Metric comparison > R2"),
        ("Figure 5.4", "MAPE comparison for all models", "section5_evaluation/assets/figure_5_4_mape_comparison.png", "Model comparison > Metric comparison > MAPE"),
        ("Figure 5.5", "Actual versus predicted prices for all models", "section5_evaluation/assets/figure_5_2_actual_vs_predicted.png", "Model comparison > Test fit and Model diagnostics > Fit"),
        ("Figure 5.6", "XGBoost residual diagnostics", "section5_evaluation/assets/figure_5_3_xgboost_residuals.png", "Model diagnostics > Residuals"),
        ("Figure 5.7", "XGBoost errors by actual price band", "section5_evaluation/assets/figure_5_4_price_band_errors.png", "Model diagnostics > Error segments"),
        ("Figure 5.8", "XGBoost permutation feature importance", "section5_evaluation/assets/figure_5_5_xgboost_permutation_importance.png", "Model diagnostics > Feature importance"),
    ],
    "Section 6 - Deployment": [
        ("Figure 6.1", "Deployment workflow", "section6_7_references/assets/figure_6_1_deployment_workflow.png", "Project overview and Price predictor"),
        ("Figure 6.2", "Streamlit project overview", "section6_deployment/assets/figure_6_1_streamlit_overview.png", "Project overview"),
        ("Figure 6.3", "Prediction controls and interactive 3D preview", "section6_deployment/assets/figure_6_2_prediction_interface.png", "Price predictor"),
        ("Figure 6.4", "Example XGBoost prediction output", "section6_deployment/assets/figure_6_3_prediction_result.png", "Price predictor"),
        ("Figure 6.5", "Interactive model-comparison dashboard", "section6_deployment/assets/figure_6_4_model_comparison.png", "Model comparison"),
    ],
}


def inject_visual_styles(page: str) -> None:
    """Apply a page-scoped dark presentation theme to the predictor only."""
    if page != "Price predictor":
        return
    st.html(
        """
        <style>
          [data-testid="stAppViewContainer"] {
            background: linear-gradient(145deg, #07111d 0%, #091522 52%, #050c15 100%);
            color: #dcebf4;
          }
          [data-testid="stHeader"] { background: rgba(5, 13, 23, .82); }
          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091522 0%, #07111c 100%);
            border-right: 1px solid rgba(112, 209, 234, .13);
          }
          [data-testid="stSidebar"] *,
          [data-testid="stAppViewContainer"] h1,
          [data-testid="stAppViewContainer"] h2,
          [data-testid="stAppViewContainer"] h3,
          [data-testid="stAppViewContainer"] p,
          [data-testid="stAppViewContainer"] label,
          [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
            color: #dcebf4;
          }
          [data-testid="stAppViewContainer"] h1 {
            letter-spacing: -.025em;
            text-shadow: 0 0 32px rgba(92, 225, 255, .12);
          }
          [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(155deg, rgba(14, 31, 47, .78), rgba(7, 18, 30, .88));
            border-color: rgba(108, 204, 229, .16) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
          }
          [data-testid="stMetric"] {
            background: rgba(8, 23, 37, .72);
            border-color: rgba(103, 216, 241, .14) !important;
          }
          [data-testid="stMetricLabel"],
          [data-testid="stMetricValue"] { color: #e7f7ff !important; }
          [data-testid="stNumberInputContainer"],
          div[data-baseweb="select"] > div {
            background: rgba(6, 19, 32, .88) !important;
            border-color: rgba(102, 200, 226, .22) !important;
          }
          [data-testid="stNumberInputContainer"] input,
          div[data-baseweb="select"] input,
          div[data-baseweb="select"] span { color: #e4f5fc !important; }
          [data-testid="stNumberInputContainer"] button {
            background: rgba(15, 40, 59, .82) !important;
            color: #c9f3ff !important;
            border-color: rgba(103, 211, 237, .13) !important;
          }
          .stButton > button[kind="primary"] {
            background: linear-gradient(100deg, #088ba7, #10b8cf) !important;
            color: #f4fdff !important;
            border: 1px solid rgba(164, 244, 255, .34) !important;
            box-shadow: 0 10px 28px rgba(4, 178, 210, .18);
          }
          [data-testid="stDataFrame"] {
            border: 1px solid rgba(103, 211, 237, .13);
            border-radius: 10px;
            overflow: hidden;
          }
          div[role="dialog"] {
            width: min(96vw, 1560px) !important;
            max-width: min(96vw, 1560px) !important;
            background: #050d17 !important;
            border: 1px solid rgba(111, 225, 247, .20) !important;
            box-shadow: 0 28px 100px rgba(0, 0, 0, .62) !important;
          }
          .st-key-diamond_result_card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(155deg, rgba(10, 27, 42, .98), rgba(4, 12, 22, .99));
            color: #e6f7ff;
            border: 1px solid rgba(112, 224, 246, .24) !important;
            box-shadow: 0 18px 55px rgba(1, 8, 18, .36), inset 0 1px 0 rgba(255,255,255,.035);
          }
          .st-key-diamond_result_card > div {
            position: relative;
            z-index: 1;
          }
          .st-key-diamond_result_card p,
          .st-key-diamond_result_card label,
          .st-key-diamond_result_card h1,
          .st-key-diamond_result_card h2,
          .st-key-diamond_result_card h3,
          .st-key-diamond_result_card [data-testid="stMetricLabel"],
          .st-key-diamond_result_card [data-testid="stMetricValue"] {
            color: #e6f7ff;
          }
        </style>
        """
    )


@st.cache_data(show_spinner=False)
def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


@st.cache_data(show_spinner=False)
def prepare_data(original_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    data = original_data.copy()
    identifier_removed = int("Unnamed: 0" in data.columns)
    if identifier_removed:
        data = data.drop(columns=["Unnamed: 0"])

    before_duplicates = len(data)
    data = data.drop_duplicates().copy()
    duplicates_removed = before_duplicates - len(data)

    before_invalid = len(data)
    data = data[(data["x"] > 0) & (data["y"] > 0) & (data["z"] > 0)].copy()
    invalid_removed = before_invalid - len(data)

    calculated_depth = 100 * data["z"] / ((data["x"] + data["y"]) / 2)
    dimension_inconsistency = (data["depth"] - calculated_depth).abs() > 10
    inconsistent_dimensions_removed = int(dimension_inconsistency.sum())
    data = data.loc[~dimension_inconsistency].copy()

    return data, {
        "identifier_removed": identifier_removed,
        "duplicates_removed": duplicates_removed,
        "invalid_removed": invalid_removed,
        "inconsistent_dimensions_removed": inconsistent_dimensions_removed,
    }


def find_dataset() -> Path | None:
    return next((path for path in DATA_FILE_CANDIDATES if path.exists()), None)


def render_brand_header() -> None:
    with st.container():
        st.markdown(":blue-badge[BMDS2003 · CRISP-DM prototype]")
        st.title("Diamond intelligence studio")
        st.markdown(
            "Explore the market structure, compare five regression models, and turn "
            "diamond characteristics into an evidence-based price estimate."
        )


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    with st.container(horizontal=True):
        for label, value, help_text in items:
            st.metric(label, value, help=help_text, border=True)


def get_model_outputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, np.ndarray]]:
    with st.status("Preparing the model laboratory…", expanded=True) as status:
        st.write("Selecting XGBoost tree count with validation-only early stopping.")
        bundle = train_model_bundle(X_train, y_train)
        st.write("Evaluating five fitted pipelines on the untouched 20% test set.")
        results, predictions = evaluate_models(bundle["models"], X_test, y_test)
        status.update(label="Five models are ready", state="complete", expanded=False)
    return bundle, results, predictions


def render_project_overview(
    df_raw: pd.DataFrame,
    df: pd.DataFrame,
    cleaning: dict[str, int],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    st.header("Project overview")
    st.caption("A rubric-aligned view of the business problem, data preparation and modelling design.")

    metric_row(
        [
            ("Raw records", f"{len(df_raw):,}", "Rows in the original CSV"),
            ("Modelling records", f"{len(df):,}", "Rows after validity checks"),
            ("Predictors", str(len(MODEL_FEATURES)), "Six numerical and three categorical"),
            ("Candidate models", "5", "One baseline and four nonlinear alternatives"),
        ]
    )

    problem_col, impact_col = st.columns(2)
    with problem_col.container(border=True, height="stretch"):
        st.subheader("Business question")
        st.write(
            "Can measurable diamond characteristics predict market price accurately enough "
            "to support rapid, consistent preliminary valuation?"
        )
    with impact_col.container(border=True, height="stretch"):
        st.subheader("Decision value")
        st.write(
            "A reliable estimate can support listing checks, inventory screening and customer "
            "conversations while keeping certified gemological appraisal as the final authority."
        )

    with st.container(border=True):
        st.subheader("CRISP-DM workflow")
        st.mermaid_chart(
            """
            flowchart LR
              A[Business understanding] --> B[Data understanding]
              B --> C[Data preparation]
              C --> D[Modelling]
              D --> E[Evaluation]
              E --> F[Deployment prototype]
              E -. refine .-> A
            """,
        )

    data_tab, preparation_tab, design_tab = st.tabs(
        ["Data understanding", "Preparation evidence", "Model design"]
    )

    with data_tab:
        data_dictionary = pd.DataFrame(
            [
                ("carat", "Numerical predictor", "Diamond weight in carats"),
                ("cut", "Categorical predictor", "Cut quality grade"),
                ("color", "Categorical predictor", "Colour grade from D to J"),
                ("clarity", "Categorical predictor", "Clarity grade from I1 to IF"),
                ("depth", "Numerical predictor", "Total depth percentage"),
                ("table", "Numerical predictor", "Table width percentage"),
                ("x, y, z", "Numerical predictors", "Physical dimensions in millimetres"),
                ("price", "Regression target", "Observed price in USD"),
            ],
            columns=["Field", "Role", "Meaning"],
        )
        st.dataframe(data_dictionary, hide_index=True, key="data_dictionary")
        st.subheader("Original data sample")
        st.dataframe(df_raw.head(20), key="raw_sample")

        summary = df[NUMERICAL_COLUMNS].describe().T.reset_index(names="Variable")
        st.subheader("Numerical summary")
        st.dataframe(
            summary,
            hide_index=True,
            key="numerical_summary",
            column_config={
                column: st.column_config.NumberColumn(format="%.2f")
                for column in summary.columns
                if column != "Variable"
            },
        )

    with preparation_tab:
        cleaning_summary = pd.DataFrame(
            [
                ("Original dataset", len(df_raw), "Preserved unchanged"),
                ("Sequential identifier", cleaning["identifier_removed"], "Excluded from modelling"),
                ("Exact analytical duplicates", cleaning["duplicates_removed"], "Removed"),
                ("Impossible dimensions (x/y/z ≤ 0)", cleaning["invalid_removed"], "Removed"),
                (
                    "Severe dimension inconsistencies",
                    cleaning["inconsistent_dimensions_removed"],
                    "Removed using reported-versus-calculated depth",
                ),
                ("Other statistical outliers", 0, "Retained unless invalid or internally inconsistent"),
                ("Final modelling dataset", len(df), "Used for train-test split"),
            ],
            columns=["Preparation step", "Rows / fields affected", "Decision"],
        )
        st.dataframe(cleaning_summary, hide_index=True, key="cleaning_summary")

        missing = df.isna().sum().rename("Missing values").reset_index(name="Missing values")
        missing = missing.rename(columns={"index": "Variable"})
        missing["Completeness (%)"] = (1 - missing["Missing values"] / len(df)) * 100
        st.subheader("Data quality check")
        st.dataframe(
            missing,
            hide_index=True,
            key="missing_values",
            column_config={
                "Completeness (%)": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )
        split_col, method_col = st.columns(2)
        with split_col.container(border=True, height="stretch"):
            st.metric("Training partition", f"{len(X_train):,} rows", "80%")
            st.caption("Used for fitting and internal validation only.")
        with method_col.container(border=True, height="stretch"):
            st.metric("Holdout test partition", f"{len(X_test):,} rows", "20%")
            st.caption("Untouched until final model evaluation; random_state=42.")

    with design_tab:
        st.dataframe(MODEL_RATIONALE, hide_index=True, key="model_rationale")
        st.info(
            "The baseline tests whether complexity is justified. Tree ensembles then test "
            "nonlinearity and interactions, while XGBoost uses validation-only early stopping "
            "to avoid choosing its tree count from the final test set.",
            icon=":material/model_training:",
        )
        st.warning(
            "The written report must still provide at least two supporting model-selection "
            "sources and at least five APA 7 references, including two academic papers.",
            icon=":material/menu_book:",
        )

    st.download_button(
        "Download cleaned modelling data",
        df.to_csv(index=False).encode("utf-8"),
        file_name="diamonds_modelling_data.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_exploration(df: pd.DataFrame) -> None:
    """Render interactive EDA while preserving the existing application structure."""
    st.header("Interactive data exploration")
    st.caption(
        "Filter the dataset, inspect distributions and relationships, and connect the live EDA "
        "directly to the figures discussed in Section 2 of the report."
    )

    # -----------------------------------------------------------------
    # Existing exploration filters
    # -----------------------------------------------------------------
    with st.container(border=True):
        st.subheader("Exploration filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        cuts = filter_col1.multiselect("Cut", CUT_ORDER, default=CUT_ORDER)
        colours = filter_col2.multiselect("Colour", COLOUR_ORDER, default=COLOUR_ORDER)
        clarities = filter_col3.multiselect("Clarity", CLARITY_ORDER, default=CLARITY_ORDER)
        range_col1, range_col2 = st.columns(2)
        price_range = range_col1.slider(
            "Observed price range (USD)",
            int(df["price"].min()),
            int(df["price"].max()),
            (int(df["price"].min()), int(df["price"].max())),
        )
        carat_range = range_col2.slider(
            "Carat range",
            float(df["carat"].min()),
            float(df["carat"].max()),
            (float(df["carat"].min()), float(df["carat"].max())),
            step=0.05,
        )

    filtered = df[
        df["cut"].isin(cuts)
        & df["color"].isin(colours)
        & df["clarity"].isin(clarities)
        & df["price"].between(*price_range)
        & df["carat"].between(*carat_range)
    ].copy()

    if filtered.empty:
        st.warning("No records match these filters. Widen one or more ranges.")
        return

    # -----------------------------------------------------------------
    # Existing summary metrics and automatic EDA insight
    # -----------------------------------------------------------------
    correlation = filtered["carat"].corr(filtered["price"])
    metric_row(
        [
            ("Visible records", f"{len(filtered):,}", f"{len(filtered) / len(df):.1%} of modelling data"),
            ("Median price", f"${filtered['price'].median():,.0f}", "Less sensitive to extreme prices"),
            ("Median carat", f"{filtered['carat'].median():.2f} ct", None),
            ("Carat–price correlation", f"{correlation:.3f}", "Pearson correlation in the filtered view"),
        ]
    )

    skewness = filtered["price"].skew()
    most_common_cut = filtered["cut"].mode().iloc[0]
    st.info(
        f"In this selection, price skewness is **{skewness:.2f}**, the most frequent cut is "
        f"**{most_common_cut}**, and carat has a **{correlation:.3f}** linear correlation with price. "
        "These are associations, not causal effects.",
        icon=":material/insights:",
    )

    # -----------------------------------------------------------------
    # Main EDA navigation
    # -----------------------------------------------------------------
    view = st.segmented_control(
        "Visual analysis",
        ["Distribution", "Carat relationship", "Category comparison", "Correlation"],
        default="Distribution",
        width="stretch",
    )

    # =================================================================
    # DISTRIBUTION
    # Covers report Sections 2.6.1, 2.6.2 and 2.6.7
    # =================================================================
    if view == "Distribution":
        hist = (
            alt.Chart(filtered)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("price:Q", bin=alt.Bin(maxbins=48), title="Observed price (USD)"),
                y=alt.Y("count():Q", title="Diamonds"),
                tooltip=[alt.Tooltip("count():Q", title="Diamonds")],
            )
            .properties(height=380, title="Price distribution")
        )

        box_sample = filtered.sample(min(12000, len(filtered)), random_state=RANDOM_STATE)
        box = (
            alt.Chart(box_sample)
            .mark_boxplot(extent=1.5, size=38)
            .encode(
                x=alt.X("cut:N", sort=CUT_ORDER, title="Cut quality"),
                y=alt.Y("price:Q", title="Observed price (USD)"),
                color=alt.Color("cut:N", sort=CUT_ORDER, legend=None),
            )
            .properties(height=380, title="Price spread by cut")
        )

        left, right = st.columns(2)
        left.altair_chart(hist, key="price_histogram")
        right.altair_chart(box, key="price_boxplot")

        st.caption(
            "Price is right-skewed, while the cut boxplot shows raw group differences. "
            "The cut comparison is descriptive because carat, colour and clarity also vary between groups."
        )

        st.divider()
        st.subheader("Distribution of numerical diamond characteristics")
        selected_numeric = st.segmented_control(
            "Numerical feature",
            ["carat", "depth", "table", "x", "y", "z"],
            default="carat",
            key="eda_numeric_distribution_selector",
        )
        numeric_distribution = (
            alt.Chart(filtered)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(f"{selected_numeric}:Q", bin=alt.Bin(maxbins=45), title=selected_numeric.title()),
                y=alt.Y("count():Q", title="Diamonds"),
                tooltip=[alt.Tooltip("count():Q", title="Diamonds")],
            )
            .properties(height=390, title=f"Distribution of {selected_numeric}")
        )
        st.altair_chart(numeric_distribution, key="selected_numeric_distribution")
        st.caption(
            "Use this chart to inspect the distributions discussed in Section 2.6.7, especially carat, depth and table."
        )

    # =================================================================
    # CARAT RELATIONSHIP
    # Covers report Sections 2.6.5 and 2.6.9
    # =================================================================
    elif view == "Carat relationship":
        scatter_sample = filtered.sample(min(6500, len(filtered)), random_state=RANDOM_STATE)
        scatter = (
            alt.Chart(scatter_sample)
            .mark_circle(opacity=0.48, size=42)
            .encode(
                x=alt.X("carat:Q", title="Carat"),
                y=alt.Y("price:Q", title="Observed price (USD)"),
                color=alt.Color("cut:N", sort=CUT_ORDER, title="Cut"),
                tooltip=[
                    alt.Tooltip("carat:Q", format=".2f"),
                    alt.Tooltip("price:Q", format="$,.0f"),
                    "cut:N",
                    "color:N",
                    "clarity:N",
                ],
            )
            .properties(height=500, title="Carat and price")
            .interactive()
        )
        st.altair_chart(scatter, key="carat_price_scatter")
        st.caption(
            f"The filtered Pearson correlation is {correlation:.3f}. The upward pattern is strong but not perfectly linear, "
            "which supports comparing nonlinear regression models with the linear baseline."
        )

        st.divider()
        left, right = st.columns(2)

        carat_hist = (
            alt.Chart(filtered)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("carat:Q", bin=alt.Bin(maxbins=45), title="Carat"),
                y=alt.Y("count():Q", title="Diamonds"),
                tooltip=[alt.Tooltip("count():Q", title="Diamonds")],
            )
            .properties(height=390, title="Carat distribution")
        )
        left.altair_chart(carat_hist, key="eda_carat_distribution")

        carat_group_data = filtered.copy()
        carat_group_data["Carat group"] = pd.cut(
            carat_group_data["carat"],
            bins=[0, 0.50, 1.00, 1.50, 2.00, np.inf],
            labels=["≤ 0.50", "0.51–1.00", "1.01–1.50", "1.51–2.00", "> 2.00"],
        )
        carat_group_sample = carat_group_data.sample(
            min(12000, len(carat_group_data)), random_state=RANDOM_STATE
        )
        carat_group_box = (
            alt.Chart(carat_group_sample)
            .mark_boxplot(extent=1.5, size=42)
            .encode(
                x=alt.X(
                    "Carat group:N",
                    sort=["≤ 0.50", "0.51–1.00", "1.01–1.50", "1.51–2.00", "> 2.00"],
                    title="Carat group",
                ),
                y=alt.Y("price:Q", title="Observed price (USD)"),
                color=alt.Color("Carat group:N", legend=None),
                tooltip=[alt.Tooltip("price:Q", format="$,.0f")],
            )
            .properties(height=390, title="Price distribution across carat groups")
        )
        right.altair_chart(carat_group_box, key="eda_price_by_carat_group")

        st.caption(
            "Prices generally rise across carat groups, but the within-group spread shows that carat alone does not determine price."
        )

    # =================================================================
    # CATEGORY COMPARISON
    # Covers report Sections 2.6.2, 2.6.3 and 2.6.4
    # =================================================================
    elif view == "Category comparison":
        dimension = st.segmented_control(
            "Compare by", ["cut", "color", "clarity"], default="cut", key="eda_category_selector"
        )
        ordering = {"cut": CUT_ORDER, "color": COLOUR_ORDER, "clarity": CLARITY_ORDER}[dimension]

        category_summary = (
            filtered.groupby(dimension, observed=False)["price"]
            .agg(["mean", "median", "count"])
            .reindex(ordering)
            .dropna()
            .reset_index()
        )

        left, right = st.columns([0.95, 1.05])

        category_chart = (
            alt.Chart(category_summary)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                y=alt.Y(f"{dimension}:N", sort=ordering, title=dimension.title()),
                x=alt.X("median:Q", title="Median observed price (USD)"),
                color=alt.Color("mean:Q", title="Mean price", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    alt.Tooltip(f"{dimension}:N", title=dimension.title()),
                    alt.Tooltip("median:Q", format="$,.0f"),
                    alt.Tooltip("mean:Q", format="$,.0f"),
                    alt.Tooltip("count:Q", format=","),
                ],
            )
            .properties(height=430, title=f"Price profile by {dimension}")
        )
        left.altair_chart(category_chart, key="category_price_chart")

        category_box_sample = filtered.sample(min(12000, len(filtered)), random_state=RANDOM_STATE)
        category_box = (
            alt.Chart(category_box_sample)
            .mark_boxplot(extent=1.5, size=36)
            .encode(
                x=alt.X(f"{dimension}:N", sort=ordering, title=dimension.title()),
                y=alt.Y("price:Q", title="Observed price (USD)"),
                color=alt.Color(f"{dimension}:N", sort=ordering, legend=None),
            )
            .properties(height=430, title=f"Price spread by {dimension}")
        )
        right.altair_chart(category_box, key="category_price_boxplot")

        st.dataframe(
            category_summary,
            hide_index=True,
            key="category_summary",
            column_config={
                "mean": st.column_config.NumberColumn("Mean price", format="$%.0f"),
                "median": st.column_config.NumberColumn("Median price", format="$%.0f"),
                "count": st.column_config.NumberColumn("Diamonds", format="localized"),
            },
        )
        st.caption(
            "These category comparisons describe associations in the observed data. They should not be interpreted as isolated causal effects."
        )

    # =================================================================
    # CORRELATION
    # Covers report Sections 2.6.6 and 2.6.8
    # =================================================================
    else:
        corr = filtered[NUMERICAL_COLUMNS].corr()
        corr_long = corr.rename_axis("Variable A").reset_index().melt(
            id_vars="Variable A", var_name="Variable B", value_name="Correlation"
        )

        heatmap = (
            alt.Chart(corr_long)
            .mark_rect(cornerRadius=2)
            .encode(
                x=alt.X("Variable A:N", sort=NUMERICAL_COLUMNS, title=None),
                y=alt.Y("Variable B:N", sort=NUMERICAL_COLUMNS, title=None),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                ),
                tooltip=["Variable A:N", "Variable B:N", alt.Tooltip("Correlation:Q", format=".3f")],
            )
            .properties(height=500, title="Numerical correlation matrix")
        )
        labels = (
            alt.Chart(corr_long)
            .mark_text(fontSize=12)
            .encode(
                x=alt.X("Variable A:N", sort=NUMERICAL_COLUMNS),
                y=alt.Y("Variable B:N", sort=NUMERICAL_COLUMNS),
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    "abs(datum.Correlation) > 0.55",
                    alt.value("white"),
                    alt.value("#0F172A"),
                ),
            )
        )

        price_corr = (
            corr["price"]
            .drop("price")
            .sort_values()
            .rename("Correlation")
            .reset_index()
)

        price_corr.columns = [
            "Feature",
            "Correlation"
        ]
        
        price_corr_chart = (
            alt.Chart(price_corr)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                y=alt.Y("Feature:N", sort="x", title=None),
                x=alt.X("Correlation:Q", title="Pearson correlation with price"),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["Feature:N", alt.Tooltip("Correlation:Q", format=".3f")],
            )
            .properties(height=500, title="Numerical correlation with price")
        )

        left, right = st.columns([1.15, 0.85])
        left.altair_chart(heatmap + labels, key="correlation_heatmap")
        right.altair_chart(price_corr_chart, key="price_correlation_chart")

        st.divider()
        st.subheader("Physical dimensions and price")
        dimension = st.segmented_control(
            "Dimension", ["x", "y", "z"], default="x", key="eda_dimension_selector"
        )
        dimension_sample = filtered.sample(min(6500, len(filtered)), random_state=RANDOM_STATE)
        dimension_chart = (
            alt.Chart(dimension_sample)
            .mark_circle(opacity=0.42, size=40)
            .encode(
                x=alt.X(f"{dimension}:Q", title=f"{dimension} dimension (mm)"),
                y=alt.Y("price:Q", title="Observed price (USD)"),
                color=alt.Color("carat:Q", title="Carat", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    alt.Tooltip(f"{dimension}:Q", format=".2f"),
                    alt.Tooltip("carat:Q", format=".2f"),
                    alt.Tooltip("price:Q", format="$,.0f"),
                    "cut:N",
                    "color:N",
                    "clarity:N",
                ],
            )
            .properties(height=460, title=f"{dimension} dimension and price")
            .interactive()
        )
        st.altair_chart(dimension_chart, key="dimension_price_scatter")
        st.caption(
            "Carat and x/y/z are strongly related measures of diamond size. Their correlations and later feature-importance values should therefore be interpreted together."
        )

    # -----------------------------------------------------------------
    # Existing filtered-record inspection remains unchanged
    # -----------------------------------------------------------------
    with st.expander("Inspect filtered records", icon=":material/table_view:"):
        st.dataframe(filtered.head(500), key="filtered_records")
        st.download_button(
            "Download filtered data",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_diamonds.csv",
            mime="text/csv",
            icon=":material/download:",
        )


def render_report_figures() -> None:
    """Expose the exact numbered outputs used in the written report."""
    st.header("Report figure library")
    st.caption(
        "Every numbered figure used in Sections 1-6 is indexed here. Select a section and "
        "figure to show the exact report output, then use the listed app location for its "
        "interactive or live equivalent."
    )

    section = st.selectbox("Report section", list(REPORT_FIGURES), key="report_figure_section")
    figures = REPORT_FIGURES[section]
    selected = st.selectbox(
        "Figure",
        figures,
        format_func=lambda item: f"{item[0]} - {item[1]}",
        key="report_figure_choice",
    )
    figure_number, title, relative_path, app_location = selected
    figure_path = APP_DIR / relative_path

    with st.container(border=True):
        st.subheader(f"{figure_number} {title}")
        if figure_path.exists():
            st.image(str(figure_path), caption=f"{figure_number} {title}", width="stretch")
        else:
            st.error(f"Report asset is missing: {relative_path}", icon=":material/broken_image:")
        st.info(f"**Interactive or live location:** {app_location}", icon=":material/ads_click:")

    index = pd.DataFrame(
        figures,
        columns=["Figure", "Report title", "Asset", "Interactive or live location"],
    ).drop(columns="Asset")
    with st.expander(f"Show the complete {section.split(' - ')[0]} figure index"):
        st.dataframe(index, hide_index=True, width="stretch", key="report_figure_index")

    st.success(
        f"{sum(len(items) for items in REPORT_FIGURES.values())} report figures are registered "
        "and can be checked from this page.",
        icon=":material/fact_check:",
    )


def render_leaderboard(
    bundle: dict[str, object],
    results: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    best = results.iloc[0]
    best_name = str(best["Model"])
    baseline = results.loc[results["Model"] == "Linear regression (baseline)"].iloc[0]

    st.success(
        f"{best_name} ranks first by test RMSE, improving on the baseline by "
        f"{best['RMSE gain vs baseline (%)']:.1f}%.",
        icon=":material/trophy:",
    )
    metric_row(
        [
            ("Test MAE", f"${best['MAE']:,.0f}", "Typical absolute error"),
            ("Test RMSE", f"${best['RMSE']:,.0f}", "Primary selection metric; penalises large misses"),
            ("Test MAPE", f"{best['MAPE (%)']:.2f}%", "Average percentage error"),
            ("Test R²", f"{best['R²']:.4f}", "Share of test-set variance explained"),
        ]
    )

    st.subheader("Holdout leaderboard")
    st.dataframe(
        results,
        hide_index=True,
        key="model_leaderboard",
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d", pinned=True),
            "MAE": st.column_config.NumberColumn(format="$%.2f"),
            "RMSE": st.column_config.NumberColumn(format="$%.2f"),
            "MAPE (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
            "RMSE gain vs baseline (%)": st.column_config.ProgressColumn(
                "RMSE gain vs baseline", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )
    st.caption(
        f"Evaluation uses {len(X_test):,} untouched test records. The linear baseline RMSE is "
        f"${baseline['RMSE']:,.2f}; lower MAE/RMSE/MAPE and higher R² are better."
    )

    view = st.segmented_control(
        "Analysis view",
        ["Metric comparison", "Test fit", "XGBoost tuning", "Model rationale"],
        default="Metric comparison",
        width="stretch",
    )

    if view == "Metric comparison":
        metric = st.segmented_control(
            "Metric", ["RMSE", "MAE", "MAPE (%)", "R²"], default="RMSE"
        )
        chart_data = results.copy()
        chart_data["Result"] = np.where(chart_data["Rank"] == 1, "Best test result", "Other model")
        metric_chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                y=alt.Y("Model:N", sort=alt.SortField(field=metric, order="ascending"), title=None),
                x=alt.X(f"{metric}:Q", title=metric),
                color=alt.Color(
                    "Result:N",
                    scale=alt.Scale(domain=["Best test result", "Other model"], range=["#22D3EE", "#334155"]),
                    legend=None,
                ),
                tooltip=["Model:N", alt.Tooltip(f"{metric}:Q", format=".3f")],
            )
            .properties(height=390, title=f"Models compared by {metric}")
        )
        st.altair_chart(metric_chart, key="model_metric_chart")

    elif view == "Test fit":
        predicted = predictions[best_name]
        diagnostics = pd.DataFrame(
            {
                "Actual price": y_test.to_numpy(),
                "Predicted price": predicted,
                "Residual": y_test.to_numpy() - predicted,
            }
        )
        sample = diagnostics.sample(min(5000, len(diagnostics)), random_state=RANDOM_STATE)
        lower = min(sample["Actual price"].min(), sample["Predicted price"].min())
        upper = max(sample["Actual price"].max(), sample["Predicted price"].max())
        equality = pd.DataFrame({"Actual price": [lower, upper], "Predicted price": [lower, upper]})
        points = (
            alt.Chart(sample)
            .mark_circle(opacity=0.42, size=38)
            .encode(
                x=alt.X("Actual price:Q", title="Actual price (USD)"),
                y=alt.Y("Predicted price:Q", title="Predicted price (USD)"),
                color=alt.Color("Residual:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=[
                    alt.Tooltip("Actual price:Q", format="$,.0f"),
                    alt.Tooltip("Predicted price:Q", format="$,.0f"),
                    alt.Tooltip("Residual:Q", format="$,.0f"),
                ],
            )
        )
        line = alt.Chart(equality).mark_line(strokeDash=[7, 5], color="#334155").encode(
            x="Actual price:Q", y="Predicted price:Q"
        )
        st.altair_chart((points + line).properties(height=500).interactive(), key="best_fit_chart")

    elif view == "XGBoost tuning":
        curve = pd.DataFrame(
            {
                "Boosting round": np.arange(1, len(bundle["validation_rmse"]) + 1),
                "Validation RMSE": bundle["validation_rmse"],
            }
        )
        learning_curve = (
            alt.Chart(curve)
            .mark_line()
            .encode(
                x=alt.X("Boosting round:Q"),
                y=alt.Y("Validation RMSE:Q", scale=alt.Scale(zero=False)),
                tooltip=["Boosting round:Q", alt.Tooltip("Validation RMSE:Q", format=".2f")],
            )
        )
        marker = alt.Chart(
            pd.DataFrame({"Boosting round": [bundle["best_n_estimators"]]})
        ).mark_rule(color="#FBBF24", strokeDash=[6, 5]).encode(x="Boosting round:Q")
        st.altair_chart(
            (learning_curve + marker).properties(height=430, title="Validation-only early stopping").interactive(),
            key="early_stopping_curve",
        )
        metric_row(
            [
                ("Best zero-based iteration", f"{bundle['best_iteration']:,}", None),
                ("Final tree count", f"{bundle['best_n_estimators']:,}", None),
                ("Patience", "50 rounds", "Training stops after 50 rounds without improvement"),
            ]
        )
        st.caption(
            "Early stopping uses only a validation split from the 80% training partition. "
            "The final 20% test set never chooses the number of trees."
        )

    else:
        st.dataframe(MODEL_RATIONALE, hide_index=True, key="model_design_table")
        with st.container(border=True):
            st.subheader("Interpretation")
            st.write(
                "The nonlinear ensembles are compared with a transparent linear baseline. "
                "A material RMSE gain supports the need for nonlinear relationships and feature "
                "interactions rather than choosing a complex model by default."
            )
        with st.container(border=True):
            st.subheader("Limitations and next improvements")
            st.write(
                "The data represents historical listed prices rather than certified valuations; "
                "external market shifts and unrecorded gem properties may affect generalisation. "
                "Future work should add cross-validation, broader hyperparameter search, temporal "
                "or external validation, calibrated prediction intervals and model monitoring."
            )

    st.download_button(
        "Download test results",
        results.to_csv(index=False).encode("utf-8"),
        file_name="diamond_model_test_results.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_diagnostics(
    bundle: dict[str, object],
    results: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    model_name = st.selectbox("Inspect a fitted model", list(bundle["models"].keys()))
    model_result = results.loc[results["Model"] == model_name].iloc[0]
    predicted = predictions[model_name]
    diagnostics = pd.DataFrame(
        {
            "Actual price": y_test.to_numpy(),
            "Predicted price": predicted,
            "Residual": y_test.to_numpy() - predicted,
        }
    )
    diagnostics["Absolute error"] = diagnostics["Residual"].abs()

    metric_row(
        [
            ("MAE", f"${model_result['MAE']:,.0f}", None),
            ("RMSE", f"${model_result['RMSE']:,.0f}", None),
            ("MAPE", f"{model_result['MAPE (%)']:.2f}%", None),
            ("R²", f"{model_result['R²']:.4f}", None),
        ]
    )
    view = st.segmented_control(
        "Diagnostic view",
        ["Fit", "Residuals", "Error segments", "Feature importance"],
        default="Fit",
        width="stretch",
    )

    sample = diagnostics.sample(min(5000, len(diagnostics)), random_state=RANDOM_STATE)
    if view == "Fit":
        lower = min(sample["Actual price"].min(), sample["Predicted price"].min())
        upper = max(sample["Actual price"].max(), sample["Predicted price"].max())
        equality = pd.DataFrame({"Actual price": [lower, upper], "Predicted price": [lower, upper]})
        points = (
            alt.Chart(sample)
            .mark_circle(opacity=0.42, size=40)
            .encode(
                x=alt.X("Actual price:Q"),
                y=alt.Y("Predicted price:Q"),
                color=alt.Color("Absolute error:Q", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    alt.Tooltip("Actual price:Q", format="$,.0f"),
                    alt.Tooltip("Predicted price:Q", format="$,.0f"),
                    alt.Tooltip("Absolute error:Q", format="$,.0f"),
                ],
            )
        )
        line = alt.Chart(equality).mark_line(color="#334155", strokeDash=[7, 5]).encode(
            x="Actual price:Q", y="Predicted price:Q"
        )
        st.altair_chart((points + line).properties(height=500).interactive(), key="diagnostic_fit")

    elif view == "Residuals":
        scatter = (
            alt.Chart(sample)
            .mark_circle(opacity=0.42, size=38)
            .encode(
                x=alt.X("Predicted price:Q", title="Predicted price (USD)"),
                y=alt.Y("Residual:Q", title="Residual: actual − predicted (USD)"),
                color=alt.condition("datum.Residual >= 0", alt.value("#38BDF8"), alt.value("#F87171")),
                tooltip=[
                    alt.Tooltip("Predicted price:Q", format="$,.0f"),
                    alt.Tooltip("Residual:Q", format="$,.0f"),
                ],
            )
            .properties(height=410, title="Residual pattern")
            .interactive()
        )
        histogram = (
            alt.Chart(diagnostics)
            .mark_bar()
            .encode(
                x=alt.X("Residual:Q", bin=alt.Bin(maxbins=55), title="Residual (USD)"),
                y=alt.Y("count():Q", title="Test records"),
                tooltip=[alt.Tooltip("count():Q")],
            )
            .properties(height=410, title="Residual distribution")
        )
        left, right = st.columns(2)
        left.altair_chart(scatter, key="residual_scatter")
        right.altair_chart(histogram, key="residual_histogram")

    elif view == "Error segments":
        diagnostics["Actual price band"] = pd.qcut(
            diagnostics["Actual price"], q=5, duplicates="drop"
        ).astype(str)
        segment = (
            diagnostics.groupby("Actual price band", observed=False)["Absolute error"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        chart = (
            alt.Chart(segment)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Actual price band:N", sort=None, title="Actual price quintile"),
                y=alt.Y("mean:Q", title="Mean absolute error (USD)"),
                color=alt.Color("mean:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=[
                    "Actual price band:N",
                    alt.Tooltip("mean:Q", format="$,.0f"),
                    alt.Tooltip("median:Q", format="$,.0f"),
                    alt.Tooltip("count:Q", format=","),
                ],
            )
            .properties(height=430, title="Where the model makes larger errors")
        )
        st.altair_chart(chart, key="error_segments")
        st.caption(
            "Price-band errors reveal whether one overall RMSE hides weaker performance for "
            "lower- or higher-value diamonds."
        )

    else:
        with st.status("Calculating permutation importance…", expanded=False) as status:
            importance = calculate_permutation_importance(
                model_name,
                bundle["models"][model_name],
                X_test,
                y_test,
            )
            status.update(label="Feature importance ready", state="complete")
        chart = (
            alt.Chart(importance)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                y=alt.Y("Feature:N", sort="-x", title=None),
                x=alt.X("Importance:Q", title="Decrease in predictive performance"),
                color=alt.Color("Importance:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".3f")],
            )
            .properties(height=430, title="Permutation feature importance")
        )
        st.altair_chart(chart, key="feature_importance")
        st.warning(
            "Importance explains how the fitted model uses a feature. It is not evidence that "
            "the feature causes a price change.",
            icon=":material/warning:",
        )


@st.dialog("Immersive diamond studio", width="large")
def render_immersive_diamond(figure: object) -> None:
    """Open the live diamond scene in a near-full-screen modal."""
    st.caption(
        "Turntable drag keeps the diamond upright. Drag left/right, scroll to zoom, "
        "or start the left-right scan from the controls below the diamond."
    )
    figure.update_layout(height=720, margin={"l": 0, "r": 0, "t": 4, "b": 52})
    st.plotly_chart(
        figure,
        key="immersive_diamond_3d",
        width="stretch",
        height=720,
        theme=None,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "scrollZoom": True,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso3d", "select2d"],
        },
    )


def render_predictor(
    bundle: dict[str, object],
    results: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    best = results.iloc[0]
    best_name = str(best["Model"])
    best_pipeline = bundle["models"][best_name]

    form_col, visual_col = st.columns([0.88, 1.12], vertical_alignment="top")
    with form_col:
        with st.container(border=True, key="predictor_specs_card"):
            st.subheader("Diamond specifications")
            st.caption("Live mode: specifications, 3D geometry and estimated price update together.")
            left, right = st.columns(2)
            left.markdown("**Quality**")
            right.markdown("**Proportions**")
            carat = left.number_input(
                "Carat",
                min_value=0.01,
                max_value=float(df["carat"].max() * 1.5),
                value=float(df["carat"].median()),
                step=0.01,
            )
            cut = left.selectbox("Cut", CUT_ORDER, index=4)
            colour = left.selectbox("Colour grade", COLOUR_ORDER, index=3)
            clarity = left.selectbox("Clarity", CLARITY_ORDER, index=3)
            depth = right.number_input(
                "Depth (%)", min_value=0.01, value=float(df["depth"].median()), step=0.1
            )
            table = right.number_input(
                "Table (%)", min_value=0.01, value=float(df["table"].median()), step=0.1
            )
            x_value = right.number_input(
                "Length x (mm)", min_value=0.01, value=float(df["x"].median()), step=0.01
            )
            y_value = right.number_input(
                "Width y (mm)", min_value=0.01, value=float(df["y"].median()), step=0.01
            )
            z_value = right.number_input(
                "Depth z (mm)", min_value=0.01, value=float(df["z"].median()), step=0.01
            )
            st.success(
                "Live valuation is active. Every input change triggers a fresh model prediction.",
                icon=":material/bolt:",
            )

        current_input = pd.DataFrame(
            {
                "carat": [carat],
                "cut": [cut],
                "color": [colour],
                "clarity": [clarity],
                "depth": [depth],
                "table": [table],
                "x": [x_value],
                "y": [y_value],
                "z": [z_value],
            }
        )
        estimate = max(0.0, float(best_pipeline.predict(current_input)[0]))
        numeric_values = {
            "carat": carat,
            "depth": depth,
            "table": table,
            "x": x_value,
            "y": y_value,
            "z": z_value,
        }
        outside = [
            name
            for name, value in numeric_values.items()
            if value < X_train[name].min() or value > X_train[name].max()
        ]
        test_absolute_errors = np.abs(y_test.to_numpy() - predictions[best_name])
        error_band = float(np.quantile(test_absolute_errors, 0.80))
        stored = {
            "estimate": estimate,
            "lower": max(0.0, estimate - error_band),
            "upper": estimate + error_band,
            "outside": outside,
            "carat": carat,
            "cut": cut,
            "colour": colour,
            "clarity": clarity,
            "input": current_input,
        }
        st.session_state["diamond_prediction"] = stored

    with visual_col:
        with st.container(border=True, key="diamond_result_card"):
            st.markdown(":blue-badge[Live WebGL turntable]")
            brilliance_col, toggle_col = st.columns([1.2, 0.8], vertical_alignment="bottom")
            brilliance_percent = brilliance_col.slider(
                "Diamond brilliance",
                min_value=20,
                max_value=100,
                value=82,
                step=2,
                key="diamond_brilliance",
                help="Adjust only the diamond's facet glow and sparkle intensity.",
            )
            show_measurements = toggle_col.toggle(
                "Measurements",
                value=True,
                key="show_3d_measurements",
                help="Show physical x, y and z dimension guides inside the 3D scene.",
            )
            diamond_figure = create_diamond_figure(
                carat=float(carat),
                colour_grade=str(colour),
                cut=str(cut),
                clarity=str(clarity),
                x_mm=float(x_value),
                y_mm=float(y_value),
                z_mm=float(z_value),
                depth_percentage=float(depth),
                table_percentage=float(table),
                show_measurements=show_measurements,
                brilliance=brilliance_percent / 100,
            )
            st.plotly_chart(
                diamond_figure,
                key="diamond_3d_view",
                width="stretch",
                height=440,
                theme=None,
                config={
                    "displaylogo": False,
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "responsive": True,
                    "modeBarButtonsToRemove": ["lasso3d", "select2d"],
                },
            )

            if st.button(
                "Open immersive diamond view",
                icon=":material/open_in_full:",
                width="stretch",
                key="open_immersive_diamond",
            ):
                render_immersive_diamond(diamond_figure)

            dimension_cols = st.columns(3, gap="small")
            dimension_cols[0].metric(
                "Face-up size", f"{x_value:.2f} × {y_value:.2f} mm", border=True
            )
            dimension_cols[1].metric("Total depth", f"{z_value:.2f} mm", border=True)
            dimension_cols[2].metric("Table", f"{table:.1f}%", border=True)

            quality_cols = st.columns(3, gap="small")
            quality_cols[0].metric("Cut geometry", cut, border=True)
            quality_cols[1].metric("Colour tint", colour, border=True)
            quality_cols[2].metric("Clarity profile", clarity, border=True)

            st.metric(
                "Live estimated market value",
                f"${stored['estimate']:,.2f}",
                help=f"Recalculated immediately by {best_name}",
            )
            st.caption(
                f"{stored['carat']:.2f} ct  ·  {stored['cut']} cut  ·  "
                f"Colour {stored['colour']}  ·  Clarity {stored['clarity']}"
            )

        st.caption(
            "Drag left or right to inspect every facet · scroll to zoom · use Left-right scan for animation. Geometry follows "
            "x/y/z, depth and table; carat is weight rather than a length measurement."
        )

    if stored["outside"]:
        st.warning(
            "Input outside the training range: " + ", ".join(stored["outside"]) + ". Treat this extrapolation cautiously.",
            icon=":material/warning:",
        )

    st.subheader("Prediction result")
    percentile = float((df["price"] <= stored["estimate"]).mean() * 100)
    metric_row(
        [
            ("Estimated price", f"${stored['estimate']:,.2f}", f"Generated by {best_name}"),
            ("Empirical 80% error band", f"${stored['lower']:,.0f} – ${stored['upper']:,.0f}", "Based on absolute errors in the holdout set"),
            ("Market position", f"{percentile:.0f}th percentile", "Relative to observed dataset prices"),
            ("Model test RMSE", f"${best['RMSE']:,.0f}", "Overall holdout performance, not case-specific error"),
        ]
    )

    input_row = stored["input"].iloc[0]
    comparable = df[
        (df["cut"] == input_row["cut"])
        & (df["color"] == input_row["color"])
        & (df["clarity"] == input_row["clarity"])
    ].copy()
    if len(comparable) < 6:
        comparable = df[(df["cut"] == input_row["cut"]) & (df["color"] == input_row["color"])].copy()
    comparable["Carat difference"] = (comparable["carat"] - input_row["carat"]).abs()
    comparable = comparable.nsmallest(8, "Carat difference")

    with st.container(border=True):
        st.subheader("Closest observed diamonds")
        st.caption("Nearest records by carat among matching quality grades; shown as context, not training labels for this input.")
        st.dataframe(
            comparable[["carat", "cut", "color", "clarity", "depth", "table", "price"]],
            hide_index=True,
            key="comparable_diamonds",
            column_config={
                "carat": st.column_config.NumberColumn(format="%.2f ct"),
                "price": st.column_config.NumberColumn(format="$%.0f"),
            },
        )

    st.warning(
        "This is a machine-learning estimate from historical listed prices. It is not a "
        "certified appraisal, sales guarantee or substitute for gemological inspection.",
        icon=":material/gavel:",
    )


dataset_file = find_dataset()
if dataset_file is None:
    st.error(
        "Dataset not found. Place `Diamonds Prices2022.csv` beside `streamlit_app.py`.",
        icon=":material/error:",
    )
    st.stop()

df_raw = load_data(str(dataset_file))
df, cleaning = prepare_data(df_raw)
X = df[MODEL_FEATURES].copy()
y = df["price"].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

with st.sidebar:
    st.markdown("## Diamond studio")
    page = st.radio("Navigation", PAGE_OPTIONS, label_visibility="collapsed")
    st.caption("BMDS2003 · Data Science")
    st.badge("Dataset ready", icon=":material/check:", color="green")
    st.metric("Modelling records", f"{len(df):,}")
    st.metric("Holdout records", f"{len(X_test):,}")
    st.caption("Models load only when a model or prediction page is opened.")

inject_visual_styles(page)
render_brand_header()

if page == "Project overview":
    render_project_overview(df_raw, df, cleaning, X_train, X_test)
elif page == "Explore data":
    render_exploration(df)
elif page == "Report figures":
    render_report_figures()
else:
    if page == "Model comparison":
        st.header("Model comparison")
        st.caption("A clearer testing output: ranked metrics, interactive fit and tuning evidence.")
    elif page == "Model diagnostics":
        st.header("Model diagnostics")
        st.caption("Inspect fit, residual structure, error segments and model-agnostic importance.")
    else:
        st.header("Diamond price predictor")
        st.caption("A presentation-ready deployment prototype with an animated valuation experience.")

    model_bundle, model_results, model_predictions = get_model_outputs(
        X_train, y_train, X_test, y_test
    )
    if page == "Model comparison":
        render_leaderboard(model_bundle, model_results, model_predictions, X_test, y_test)
    elif page == "Model diagnostics":
        render_diagnostics(model_bundle, model_results, model_predictions, X_test, y_test)
    else:
        render_predictor(
            model_bundle,
            model_results,
            model_predictions,
            df,
            X_train,
            y_test,
        )

st.space("large")
st.caption(
    "Diamond intelligence studio · Reproducible 80:20 split · Random state 42 · "
    "Educational prototype for BMDS2003"
)

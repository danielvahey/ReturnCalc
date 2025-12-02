"""
Daniel Vahey
"""

import streamlit as st
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


def compute_irr(cashflows, dates):
    """Compute IRR using actual day count with Newton's method."""
    # Convert dates to year fractions
    t = (dates - dates.iloc[0]).dt.days / 365.25

    def npv(rate):
        return np.sum(cashflows / (1 + rate) ** t)

    # Newton iteration
    rate = 0.1
    for _ in range(100):
        f = npv(rate)
        # derivative approximation
        d = (npv(rate + 1e-6) - f) / 1e-6
        rate -= f / d
    return rate


def compute_time_weighted_index_value(pe_df, index_df):
    """
    Computes time-weighted benchmark return using index levels at
    private fund cashflow dates.
    """

    # Merge index values to cashflow dates
    df = pd.merge_asof(
        pe_df[['date']].sort_values("date"),
        index_df.sort_values("date"),
        on="date",
        direction="backward"
    )

    df['index_value'] = df['index_value'].ffill().bfill()

    # compute period returns
    df['index_return'] = df['index_value'].pct_change()

    # chain returns
    twrr = np.prod(1 + df['index_return'].iloc[1:]) - 1

    return twrr


def compute_direct_alpha(pe_df, index_df):
    """
    Computes direct alpha using modified public-equivalent scaling.

    pe_df must contain: date, cashflow, nav
    index_df must contain: date, index_value
    """

    df = pd.merge_asof(
        pe_df.sort_values("date"),
        index_df.sort_values("date"),
        on="date",
        direction="backward"
    )

    df['index_value'] = df['index_value'].ffill().bfill()

    final_index = df['index_value'].iloc[-1]

    # public-scaled equivalent cashflows
    df['public_cf'] = df['cashflow'] * (final_index / df['index_value'])
    df['public_cf'] += df['nav'] * (final_index / df['index_value'])

    # Compute IRRs
    irr_private = compute_irr(df['cashflow'], df['date'])
    irr_public_equiv = compute_irr(df['public_cf'], df['date'])

    direct_alpha = np.log1p(irr_private) - np.log1p(irr_public_equiv)

    return irr_private, irr_public_equiv, direct_alpha


def generate_pdf_report(
    mPME,
    theoretical_value,
    actual_value,
    irr_private,
    irr_public,
    direct_alpha,
    twrr,
    output_path="/tmp/pe_report.pdf"
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=letter)

    story = []

    def add(text):
        story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 12))

    add("<b>Private Markets Performance Report</b>")
    add(f"Modified PME: {mPME:.4f}")
    add(f"Buy-and-Hold Index Benchmark: Theoretical = ${theoretical_value:,.0f}, Actual = ${actual_value:,.0f}")
    add(f"IRR (Private): {irr_private:.4%}")
    add(f"IRR (Public-Equivalent): {irr_public:.4%}")
    add(f"Direct Alpha: {direct_alpha:.4f}")
    add(f"Time-Weighted Public Index Return: {twrr:.4%}")

    doc.build(story)
    return output_path


st.title("Private Markets vs Public Index — PME & Buy-and-Hold Benchmark")

st.markdown("""
Upload:
1. **Private Equity cashflows + NAVs** (CSV)  
2. **Public index values** (CSV)

### Expected formats:

#### Private Equity CSV
| date | cashflow | nav |
|------|----------|-----|
| 2015-03-31 | -1000000 | 0 |
| 2016-03-31 | -250000  | 0 |
| 2019-09-30 | 600000   | 300000 |

- Cash outflows (calls) are **negative**  
- Distributions are **positive**

#### Public Index CSV
| date | index_value |
|------|-------------|
| 2015-03-31 | 1000 |
| 2016-03-31 | 1100 |
""")

# ------------------------------
# File Upload
# ------------------------------
pe_file = st.file_uploader("Upload Private Equity Cashflows CSV", type="csv")
index_file = st.file_uploader("Upload Public Index CSV", type="csv")

if pe_file and index_file:

    # ------------------------------
    # Load and clean data
    # ------------------------------
    pe_df = pd.read_csv(pe_file)
    index_df = pd.read_csv(index_file)

    pe_df['date'] = pd.to_datetime(pe_df['date'])
    index_df['date'] = pd.to_datetime(index_df['date'])

    pe_df = pe_df.sort_values("date")
    index_df = index_df.sort_values("date")

    st.subheader("Uploaded Private Equity Cashflows")
    st.dataframe(pe_df)

    st.subheader("Uploaded Public Index Data")
    st.dataframe(index_df)

    # ------------------------------
    # Merge index values onto PE cashflow dates
    # ------------------------------
    merged = pd.merge_asof(
        pe_df.sort_values("date"),
        index_df.sort_values("date"),
        on="date",
        direction="backward"
    )

    merged.rename(columns={"index_value": "index"}, inplace=True)
    merged['index'] = merged['index'].ffill().bfill()

    # ------------------------------
    # Modified PME Calculation
    # ------------------------------
    final_index = merged['index'].iloc[-1]
    merged['growth_factor'] = final_index / merged['index']
    merged['scaled_cashflow'] = merged['cashflow'] * merged['growth_factor']
    merged['scaled_nav'] = merged['nav'] * merged['growth_factor']

    total_scaled_in = -merged.loc[merged['scaled_cashflow'] < 0, 'scaled_cashflow'].sum()
    total_scaled_out = merged.loc[merged['scaled_cashflow'] > 0, 'scaled_cashflow'].sum() + merged['scaled_nav'].iloc[-1]

    mPME = total_scaled_out / total_scaled_in

    st.header("Modified PME")
    st.metric("Modified PME", f"{mPME:.3f}")

    # ------------------------------
    # BUY-AND-HOLD PUBLIC INDEX BENCHMARK
    # ------------------------------

    st.header("Buy-and-Hold Benchmark (All Capital Calls Invested on Day 1)")

    # Step 1: Total capital called
    total_contributed = -pe_df.loc[pe_df['cashflow'] < 0, 'cashflow'].sum()

    first_call_date = pe_df.loc[pe_df['cashflow'] < 0, 'date'].min()
    last_dist_date = pe_df.loc[pe_df['cashflow'] > 0, 'date'].max()

    # Get index at first call & last distribution
    first_idx = index_df.loc[index_df['date'] <= first_call_date, 'index_value'].iloc[-1]
    last_idx = index_df.loc[index_df['date'] <= last_dist_date, 'index_value'].iloc[-1]

    # Step 2: Growth ratio
    growth_factor_buy_hold = last_idx / first_idx

    # Step 3: Theoretical value if all calls were invested at first call
    theoretical_value = total_contributed * growth_factor_buy_hold

    # Step 4: Actual private market outcome
    total_distributions = pe_df.loc[pe_df['cashflow'] > 0, 'cashflow'].sum()
    ending_nav = pe_df['nav'].iloc[-1]
    actual_value = total_distributions + ending_nav

    # ------------------------------
    # Display results
    # ------------------------------
    st.subheader("Benchmark Results")

    col1, col2 = st.columns(2)
    col1.metric("Total Capital Called", f"${total_contributed:,.0f}")
    col2.metric("Index Growth Factor", f"{growth_factor_buy_hold:.3f}")

    col3, col4 = st.columns(2)
    col3.metric("Theoretical Public Index Value\n(Invested on Day 1)", f"${theoretical_value:,.0f}")
    col4.metric("Actual PE Value\n(Distributions + NAV)", f"${actual_value:,.0f}")

    st.markdown("---")

    st.subheader("Interpretation")
    st.markdown("""
    - **If Actual Value > Theoretical Value:**  
      The private fund outperformed simply putting all capital into the index at the first call.

    - **If Actual Value < Theoretical Value:**  
      The public index provided a better buy-and-hold return.
    """)

    # ------------------------------
    # Plot for visual comparison
    # ------------------------------
    st.subheader("Comparison Chart")

    comparison_df = pd.DataFrame(
        {
            "metric": ["Theoretical Public Index Value", "Actual PE Value"],
            "value": [theoretical_value, actual_value],
        }
    )

    st.bar_chart(comparison_df.set_index("metric"))

    # ---- Direct Alpha ----
    irr_private, irr_public_equiv, direct_alpha = compute_direct_alpha(pe_df, index_df)

    st.header("Direct Alpha")
    colA, colB, colC = st.columns(3)
    colA.metric("IRR (Private)", f"{irr_private:.2%}")
    colB.metric("IRR (Public-Equivalent)", f"{irr_public_equiv:.2%}")
    colC.metric("Direct Alpha", f"{direct_alpha:.4f}")

    # ---- Time Weighted Return Benchmark ----
    twrr = compute_time_weighted_index_value(pe_df, index_df)

    st.header("Time-Weighted Public Index Benchmark")
    st.metric("TWRR", f"{twrr:.2%}")

    # ---- PDF Report Export ----
    st.header("Export Report")

    if st.button("Generate PDF"):
        pdf_path = generate_pdf_report(
            mPME,
            theoretical_value,
            actual_value,
            irr_private,
            irr_public_equiv,
            direct_alpha,
            twrr
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Report PDF",
                data=f,
                file_name="private_markets_report.pdf",
                mime="application/pdf"
            )


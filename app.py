import streamlit as st

st.set_page_config(page_title="ClinsightAI", layout="wide")
st.title("🏥 ClinsightAI — Healthcare Review Intelligence")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Theme Discovery", 
    "📈 Rating Impact", 
    "⚠️ Systemic Issues", 
    "🗺️ Action Roadmap"
])

with tab1:
    # bar chart, pie chart, network graph
    st.plotly_chart(theme_freq_fig)
    st.plotly_chart(network_fig)

with tab2:
    # regression coefficients, SHAP plot
    st.dataframe(impact_df)
    st.pyplot(shap_fig)

with tab3:
    # risk scores, isolated/recurring/systemic table
    st.dataframe(risk_df)

with tab4:
    # GPT roadmap rendered as cards
    for item in roadmap:
        st.metric(label=item['recommendation'], value=item['expected_rating_lift'])
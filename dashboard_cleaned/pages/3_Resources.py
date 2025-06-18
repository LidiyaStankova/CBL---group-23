import streamlit as st

st.set_page_config(page_title="Resources", layout="wide")
st.title("Resources")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Data Sources
    - [IoD 2019 - Deprivation Data](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)
    - Census 2021 - Population Data
        - [Age on arrival in UK](https://www.ons.gov.uk/datasets/TS018/editions/2021/versions/3)
        - [Country of birth](https://www.data.gov.uk/dataset/37a15fe5-53b4-4e42-9e56-fe7912f514ff/census-2021-country-of-birth)
        - [Economic activity](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/economicactivitystatusenglandandwales/census2021)
    - [HM LR - Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
    - [ONS - Nature of crime: burglary](https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice/datasets/natureofcrimeburglary)
    - [UK Police](https://data.police.uk/data/)
    - [London Datastore](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london/)
    """)

with col2:
    st.markdown("""
    ### Libraries & Tools
    - [Streamlit](https://streamlit.io/)
    - [Pandas](https://pandas.pydata.org/)
    - [NumPy](https://numpy.org/)
    - [Matplotlib](https://matplotlib.org/)
    - [Folium](https://python-visualization.github.io/folium/)
    - [cvxpy](https://www.cvxpy.org/)
    - [GeoPandas](https://geopandas.org/)
    - [Branca](https://branca.readthedocs.io/en/latest/)
    - [PyArrow](https://arrow.apache.org/)
    - [Shapely](https://shapely.readthedocs.io/en/stable/)
    """)
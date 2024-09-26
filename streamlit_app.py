import streamlit as st

lab1 = st.Page("LAB-01.py",title="Lab 1")
lab2 = st.Page("LAB-02.py",title="Lab 2")
lab3 = st.Page("LAB-03.py",title="Lab 3")
lab4 = st.Page("LAB-04.py",title="Lab 4")
lab5 = st.Page("LAB-05.py",title="Lab 5", default=True)
pg = st.navigation([lab1, lab2, lab3, lab4, lab5])
st.set_page_config(page_title="LAB Main Page")
pg.run()
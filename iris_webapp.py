import streamlit as st
import pickle

lin_model = pickle.load(open("lin_model.pkl", "rb"))
log_model = pickle.load(open("log_model.pkl", "rb"))
svm = pickle.load(open("svm.pkl", "rb"))


def classificar(num):
    if num < 0.5:
        return "Setosa"
    elif num < 1.5:
        return "Versicolor"
    else:
        return "Virginica"


def main():

    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Classificação de Flores do Tipo Íris</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    atividades = ["Regressão Linear", "Regressão Logística", "SVM"]
    opcao = st.sidebar.selectbox("Qual modelo você gostaria de usar?", atividades)
    st.subheader(opcao)

    sl = st.slider("Selecione o Comprimento da Sépala", 0.0, 10.0)
    sw = st.slider("Selecione a Largura da Sépala", 0.0, 10.0)
    pl = st.slider("Selecione o Comprimento da Pétala", 0.0, 10.0)
    pw = st.slider("Selecione a Largura da Pétala", 0.0, 10.0)
    inputs = [[sl, sw, pl, pw]]

    if st.button("Classificar"):
        if opcao == "Regressão Linear":
            st.success(classificar(lin_model.predict(inputs)))
        elif opcao == "Regressão Logística":
            st.success(classificar(log_model.predict(inputs)))
        else:
            st.success(classificar(svm.predict(inputs)))


if __name__ == "__main__":
    main()

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
load_dotenv(override=True)
from utilities.bankingAgent import BankingAgent

#--------------------------------------#
# MAIN                                 #
#--------------------------------------#

def main():
    if 'agent' not in st.session_state:
        st.session_state['agent'] = BankingAgent()
    agent = st.session_state['agent']

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["こんにちは！ご要件はなんでしょうか？"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["こんにちは！"]
        
    response_container = st.container()
    container = st.container()


    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("ご質問:", placeholder="ご要件をお伺いいたします。", key='input')
            submit_button = st.form_submit_button(label='送信')
            
        if submit_button and user_input:
            with st.spinner("データを処理中です..."):
                responses = agent.operator_chat(user_input)
            st.session_state['past'].append(user_input)
            # Combine assistant responses into a single string for display
            assistant_text = "\n\n".join([resp.content for resp in responses if resp and resp.name])
            st.session_state['generated'].append(assistant_text)


    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))



if __name__ == '__main__':
    main()


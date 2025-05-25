import os, json
from IPython.display import Image
import openai
import streamlit as st
from streamlit_chat import message
from azure.core.credentials import AzureKeyCredential
import asyncio
import logging
from typing import Annotated
from semantic_kernel.agents import (
    ChatCompletionAgent,
    ChatHistoryAgentThread,
    AgentGroupChat
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.agents.strategies import TerminationStrategy, KernelFunctionTerminationStrategy, KernelFunctionSelectionStrategy
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.contents import ChatHistoryTruncationReducer

from dotenv import load_dotenv
load_dotenv(override=True)

def build_chat() -> AgentGroupChat:
    #--------------------------------------#
    # DEFINE Plugins                       #
    #--------------------------------------#

    class verifyIdentity:
        """本人確認をするプラグインです"""

        @kernel_function(description="銀行カード番号をキーとして顧客マスターの情報を取得します")
        def get_clientMaster_by_bankCardNumber(
            self, bankCardNumber: Annotated[str, "銀行カード番号"]
        ) -> Annotated[str, "顧客マスターから、氏名、住所、電話番号を返します"]:
            return  """
            マイクロ太郎
            港区港南 2-16-3
            03-4332-5300
            """

    class addressUpdate:
        """住所変更をするプラグインです"""

        @kernel_function(description="銀行カード番号をキーとして顧客マスターの住所変更をします")
        def update_clientMaster_by_bankCardNumber(
            self, bankCardNumber: Annotated[str, "銀行カード番号"], newAddress: Annotated[str, "新しい住所"]
        ) -> Annotated[str, "顧客マスターから、氏名、住所、電話番号を返します"]:
            return  """
            マイクロ太郎
            港区港南 9-86-7
            03-4332-5300
            """
    #--------------------------------------#
    # ROLE DEFINITIONS of AGENTS           #
    #--------------------------------------#

    # 各エージェントレベルのインストラクションで終了条件を指定する必要はありません。
    RECEPTIONIST_NAME = "Receptionist"
    RECEPTIONIST_INSTRUCTIONS = """
    あなたは、銀行の受付係で、顧客からのリクエストを受付け、完了するまでのプロセスを管理します。
    あなたの目標は、顧客からリクエストをヒアリングし、タスクを適切な担当者に振り分け、顧客のリクエストが解決したかどうかを確認することです。
    現在銀行が行えるタスクは住所変更のみです。

    あなたは、顧客からのリクエストを受け取ったら、まず本人確認を行うために、顧客に銀行カード番号を聞いてください。
    銀行カード番号をキーとして、顧客マスターの情報を取得し、顧客の氏名、住所、電話番号を返し、正しいかどうかを確認してください。

    """

    ADDRESSUPDATER_NAME = "AddressUpdater"
    ADDRESSUPDATER_INSTRUCTIONS = """
    あなたは顧客の住所変更をする担当者です。
    あなたのタスクは、顧客から住所変更に必要な書類を提示してもらい、顧客マスターに登録されている顧客の住所を変更することです。
    まず、顧客から住所変更に必要な書類を提示してもらってください。
    タスクが完了したら、更新された顧客マスターの氏名、住所、電話番号を返してください。
    住所変更に必要な書類は次のいずれかです。
    - 運転免許証
    - 健康保険証

    """

    OTHERTASKOPERATOR_NAME = "OtherTaskOperator"
    OTHERTASKOPERATOR_INSTRUCTIONS = """
    あなたはその他のタスクを担当する担当者です。
    あなたは現在何も担当していません。
    リクエストが来たら、何もせずに「おやすみなさい」と返してください。
    """

    #--------------------------------------#
    # DEFINE AGENTS                        #
    #--------------------------------------#
    # このシナリオならカーネルは共通でも良い
    def _create_kernel_with_chat_completion(service_id: str) -> Kernel:
        kernel = Kernel()
        kernel.add_service(AzureChatCompletion(service_id=service_id, deployment_name="gpt-4o"))
        return kernel

    agent_receptionist = ChatCompletionAgent(
        kernel=_create_kernel_with_chat_completion(RECEPTIONIST_NAME),
        name=RECEPTIONIST_NAME,
        instructions=RECEPTIONIST_INSTRUCTIONS,
        plugins=[verifyIdentity()],
    )

    agent_addressupdater = ChatCompletionAgent(
        kernel=_create_kernel_with_chat_completion(ADDRESSUPDATER_NAME),
        name=ADDRESSUPDATER_NAME,
        instructions=ADDRESSUPDATER_INSTRUCTIONS,
        plugins=[addressUpdate()],
    )

    agent_othertaskoperator = ChatCompletionAgent(
        kernel=_create_kernel_with_chat_completion(OTHERTASKOPERATOR_NAME),
        name=OTHERTASKOPERATOR_NAME,
        instructions=OTHERTASKOPERATOR_INSTRUCTIONS,
    )


    #--------------------------------------#
    # KERNEL FUNCTION STRATEGY             #
    #--------------------------------------#

    # 完了条件のプロンプトでどういう結果を返すか指示します。
    # この指示が result_parser で判定されます。
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt="""
        顧客のリクエストが完了したか、もしくは顧客に質問するためにいったん会話を中断するかどうか判断します。
        顧客のリクエストが完了した場合は "<request_completed>" を返してください。
        顧客のリクエストが完了していない場合は "<return_to_user>" を返してください。
        それ以外の場合はエージェントの対応が引き続き必要です。

        History:
        {{$history}}
        """,
    )

    # 選択条件である程度のプロセスを書いた方が良さそうです
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
        会話の次の順番に進む参加者を、最新の参加者に基づいて決定します。
        次のターンに進む参加者の名前だけを述べます。
        参加者は連続して1ターン以上を取ってはいけません。
        
        次の参加者からのみ選択してください。
        - {RECEPTIONIST_NAME}
        - {ADDRESSUPDATER_NAME}
        - {OTHERTASKOPERATOR_NAME}
        
        次の参加者を選択するときは、常に次のルールに従ってください:
        - 初めは {RECEPTIONIST_NAME} から始めます。
        - ユーザーのリクエストに応じて、次の参加者を{RECEPTIONIST_NAME}以外から選択してください。
        ‐ 住所変更は {ADDRESSUPDATER_NAME} に振り分けます。
        - それ以外のリクエストは {OTHERTASKOPERATOR_NAME} に振り分けます。
        - 各担当者のタスクが完了したら一旦 {RECEPTIONIST_NAME} に対応を戻します。

        History:
        {{{{$history}}}}
        """,
    )


    #--------------------------------------#
    # INSTANTIATE AgentGroupChat           #
    #--------------------------------------#
    chat = AgentGroupChat(
        agents=[
            agent_receptionist,
            agent_addressupdater,
            agent_othertaskoperator,
        ],
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[
                agent_receptionist,
                agent_addressupdater,
                agent_othertaskoperator,
            ],
            function=termination_function,
            kernel=_create_kernel_with_chat_completion("termination"),
            result_parser=lambda result: any(keyword in str(result.value[0]).lower() for keyword in ["<return_to_user>", "<request_completed>"]),
            history_reducer=ChatHistoryTruncationReducer(target_count=1),
            history_variable_name="history",
            maximum_iterations=50,
        ),
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=agent_receptionist,
            function=selection_function,
            kernel=_create_kernel_with_chat_completion("selection"),
            result_parser=lambda result: str(result.value[0]) if result.value is not None else RECEPTIONIST_NAME,
            history_reducer=ChatHistoryTruncationReducer(target_count=1),
            history_variable_name="history",
        ),
    )

    return chat

#--------------------------------------#
# MAIN                                 #
#--------------------------------------#

def operator_chat(user_input):
    # Retrieve the AgentGroupChat instance from session state
    chat = st.session_state['chat']
    async def _run_and_collect(user_input):

        await chat.add_chat_message(message=user_input)
        responses = []
        async for resp in chat.invoke():
            responses.append(resp)
        return responses

    try:
        result = asyncio.run(_run_and_collect(user_input))

        for response in result:
            if response is None or not response.name:
                continue
            print(f"# {response.name.upper()}:\n{response.content}")

    except Exception as e:
        print(f"Error during chat invocation: {e}")

    # Reset the chat's complete flag for the new conversation round.
    chat.is_complete = False
    return result


def main():
      
    if 'chat' not in st.session_state:
        st.session_state['chat'] = build_chat()
    else:
        chat = st.session_state['chat']

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
                responses = operator_chat(user_input)
            st.session_state['past'].append(user_input)
            # Combine assistant responses into a single string for display
            assistant_text = "\n\n".join([resp.content for resp in responses if resp and resp.name])
            st.session_state['generated'].append(assistant_text)
            st.session_state['chat'] = chat



    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))



if __name__ == '__main__':
    main()


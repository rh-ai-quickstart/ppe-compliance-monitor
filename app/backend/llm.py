import os
from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a terse workplace safety assistant for a PPE compliance monitoring system. "
    "Answer ONLY from the detection context provided. Respond in 1-3 short sentences max.\n\n"
    "Scope (reject anything else with a one-line refusal):\n"
    "• Worker/people counts\n"
    "• Hardhat compliance (counts & rates)\n"
    "• Safety vest compliance (counts & rates)\n"
    "• Overall PPE compliance\n"
    "• Brief safety summaries & recommendations\n\n"
    "Rules:\n"
    "1. Use only the provided context — never speculate.\n"
    "2. If data is missing, say so in one sentence.\n"
    "3. Prefer numbers and percentages over prose.\n"
    "4. No greetings, filler words."
)


class LLMChat:
    """Conversational LLM backed by a VLLM-served OpenAI-compatible endpoint.

    Maintains per-session chat history so the model sees the full conversation.
    """

    def __init__(self) -> None:
        endpoint = os.environ["OPENAI_API_ENDPOINT"]
        api_key = os.environ["OPENAI_API_TOKEN"]
        model = os.getenv("OPENAI_MODEL", "llama-4-scout-17b-16e-w4a16")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

        llm = ChatOpenAI(
            base_url=endpoint,
            api_key=api_key,
            model=model,
            temperature=temperature,
            streaming=True,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Current detection context:\n{context}\n\nQuestion: {input}"),
            ]
        )

        self._sessions: dict[str, InMemoryChatMessageHistory] = {}
        self._chain = RunnableWithMessageHistory(
            prompt | llm,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        log.info(f"LLMChat initialised — endpoint={endpoint}, model={model}")

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
        return self._sessions[session_id]

    def ask_question(
        self,
        question: str,
        context: str,
        session_id: str = "default",
    ) -> str:
        """Send a question with context through the conversational chain.

        Every prior exchange in *session_id* is automatically included so the
        model can reference earlier questions and answers.
        """
        response = self._chain.invoke(
            {"input": question, "context": context},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content

    def stream_question(
        self,
        question: str,
        context: str,
        session_id: str = "default",
    ) -> Generator[str, None, None]:
        """Stream answer tokens one chunk at a time.

        Conversation history is updated automatically once the full stream
        has been consumed.
        """
        for chunk in self._chain.stream(
            {"input": question, "context": context},
            config={"configurable": {"session_id": session_id}},
        ):
            if chunk.content:
                yield chunk.content

    def clear_history(self, session_id: str = "default") -> None:
        """Clear the conversation history for a session."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()

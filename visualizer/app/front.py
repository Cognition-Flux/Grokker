# %%
import os
import logging

import streamlit as st
from dotenv import load_dotenv
from plotly.io import from_json

# from viz import generar_grafico
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
import sys
from datetime import datetime, date
from langchain_core.tools import tool

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_generator import (
    ChatHistory,
    initialize_model,
    instantiate_model_with_prompt_and_PlotlySchema,
    llm_json_to_plot_from_text,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)


DEBUG_MODE = False
DATE_FORMAT = "%d/%m/%Y"
FECHA_DE_HOY = datetime.now().strftime(DATE_FORMAT)
OFICINAS = [
    "Las Condes", "Providencia", "Santiago Centro", "Maipú", "Puente Alto"
    ]


# %%
def get_default_tab_data():
    """Return default data structure for a tab."""
    return {
        "name": "Chat",
        "id": None,
        "fig_json": None,
        "edit_mode": False,
        "chat_history_for_plot_gen": None,
        "llm_plot_gen": None,
    }


def initialize_session_state():
    """Initialize session state variables."""
    if "tabs" not in st.session_state:
        st.session_state.tabs = [get_default_tab_data()]
    if "tab_counter" not in st.session_state:
        st.session_state.tab_counter = 1


@st.cache_data(show_spinner=False)
def load_oficinas(minimo_dias_registro=120):
    """Loads valid offices."""
    return OFICINAS  # oficinas_con_datos_validos(minimo_dias_registro)


initialize_session_state()
model = initialize_model()


def main_tab():
    """Content and logic for the main chat tab."""

    # Initialize session state variables if they don't exist
    if "previous_selection" not in st.session_state:
        st.session_state["previous_selection"] = None
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "resultados_intermedios" not in st.session_state:
        st.session_state["resultados_intermedios"] = ""

    if "ultimo_mensaje_del_llm" not in st.session_state:
        st.session_state["ultimo_mensaje_del_llm"] = None

    # Load oficinas

    oficinas_list = load_oficinas()

    # Create a multiselect for selecting offices
    selected_elements = st.multiselect("Seleccione oficinas:", oficinas_list)

    if st.session_state["previous_selection"] != selected_elements:
        # Clear histories
        # st.session_state["history"] = []
        st.session_state["chat_history"] = []
        st.session_state["resultados_intermedios"] = ""
        # Update previous selection
        st.session_state["previous_selection"] = selected_elements.copy()
        st.session_state["ultimo_mensaje_del_llm"] = None

    if selected_elements:
        st.info(f"📍 {len(selected_elements)} oficinas seleccionadas")
        # llm = AzureChatOpenAI(
        #     azure_deployment="gpt-4o",
        #     api_version="2024-09-01-preview",
        #     temperature=0,
        #     # memory=memory,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=5,
        #     api_key=os.environ.get("AZURE_API_KEY"),
        #     azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        #     streaming=True,
        # )
        llm = initialize_model()
        # Display the conversation history
        for message in st.session_state["history"]:
            with st.chat_message(message["role"]):
                st.markdown(
                    f'<div class="chat-message">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # Initialize user_message to None
        user_message = None

        # Get user input
        if prompt := st.chat_input(" ⌨ Escribe tu consulta aquí...", key="chat_input"):
            user_message = prompt

        # Check if a common question was clicked
        elif "clicked_question" in st.session_state:
            user_message = st.session_state["clicked_question"]
            del st.session_state["clicked_question"]  # Remove it after processing
        with st.spinner("Trabajando..."):
            # Process the user's message
            if user_message is not None:
                # Add the user's message to the history
                st.session_state["history"].append(
                    {"role": "user", "content": user_message}
                )
                # Display the user's message
                with st.chat_message("user"):
                    st.markdown(
                        f'<div class="chat-message">{user_message}</div>',
                        unsafe_allow_html=True,
                    )

                # Placeholder for the assistant's response
                with st.chat_message("assistant"):
                    st.session_state["chat_history"] = (
                        st.session_state["chat_history"]
                        + [
                            HumanMessage(
                                content=str(
                                    st.session_state["history"][0][
                                        "content"
                                    ]  # extraer el único mensaje
                                    if len(st.session_state["history"])
                                    == 1  # cuando es el primer mensaje del chat
                                    else st.session_state["history"][-2][
                                        "content"
                                    ]  # Si no el penúltimo, por ya habría un dialogo.
                                )
                            ),
                            # (str(resultados_intermedios)),
                            AIMessage(
                                content=str(
                                    None  # nada, porq si es el primer mensaje la ia todavía no responde.
                                    if len(st.session_state["history"]) == 1
                                    else st.session_state["history"][-1][
                                        "content"
                                    ]  # si no extrar el último, que siempre es un mss de la ia.
                                )
                            ),
                        ]
                    )

                    formatted_prompt = {
                        "input": user_message,
                        "chat_history": st.session_state["chat_history"],
                    }

                    system_prompt = f"""
                    
                    Oficina seleccionadas:        {", ".join([f"'{s}'" for s in selected_elements])}  
                
                    No puedes responder absolutamente ninguna pregunta sobre temáticas que no sean las de tu rol, bajo ninguna circunstancia.

                    Tu rol es: tú eres un agente que ayuda al usuario a extraer, consolidar y analizar información específica para todas las oficinas {", ".join([f"'{s}'" for s in selected_elements])} sobre atención a clientes (sin importar la historia de mensajes), series, niveles de servicio de sucursales y desempeño de ejecutivos, también puedes revisar y mostrar la historia de mensajes del chat para ayudar al usario/humano.
                    Comunícate con el usuario/humano amablemente, puedes guiarlo sobre qué información puedes proporcionarle.
                    

                    Tus respuestas tienen que ser, breves, precisas y exclusivamente pertinentes a la solicitud del usuario/humano, siempre reportando info de todas las oficinas  {", ".join([f"'{s}'" for s in selected_elements])}  (sin importar la historia de mensajes). 
                    Simpre Acota/limita/enfoca tu respuesta/análisis únicamente a lo que el usuario está preguntando.
                    
                    Cuando explicitamente se te soliciten resúmenes de resultados ("Resuma los resultados"),  única y exclusivamente debes resumir los mensajes de la tuyos de IA (AIMessage) anteriores que se han generado solamente durante la conversación, puedes sintetizar estos mensajes en tablas.
                    Tu respuesta debe ser consistente con la cronología del historial de mensajes; siempre continua el hilo de la conversación.
                    Siempre debes incluir un párrafo de análisis (sin título) de los resultados en todas tus respuestas.
                    Realiza cálculos matemáticos directamente; eres experto en matemática y estadística, y estudios comparativos complejos. Solo mostrar los resultados finales, si excepcionalmente muestras ecuaciones o pequeñas fórmulas o símbolos matemáticos, asegúrate siempre de encerrarlas entre dobles signos de dólar $$...$$ para que se muestren correctamente ($$...$$).

                    Importante: Por default, use la fecha del día de hoy  {FECHA_DE_HOY} como inicio y final (nunca preguntar, solo ejecutar directamente), siempre muestre las fechas en formato {DATE_FORMAT}
                    Importante: Si te preguntan como obtuviste y de donde sacaste los datos, sólo y únicamamente debes decir que tienes la capacidad de explorar las bases datos de las oficinas, nada más. 
                    Cuando le pregunten como calculó el nivel de servicio (SLA) indique que es el porcentaje de clientes q esperaron menos  del tiempo de espera de corte (umbral) que utilizó.
                    When calling tools/functions the parameters/arguments should be passed into a serialized JSON formatted string.
                    cuando le soliciten "Atenciones diarias del último mes", entrega los datos día a día solo son el total diario de atenciones.
                    Siempre construya una sola/única tabla con un breve parráfo de análisis.
                    {f"DO NOT CALL/INVOKE A TOOL AGAIN IF THE NEEDED INFO IS ALREADY HERE IN THE FOLLOWING (if so, just use this info and directly respond to the user/human): {st.session_state['resultados_intermedios']}" if st.session_state["resultados_intermedios"].__len__() > 0 else ''}
                    """
                    if DEBUG_MODE:
                        st.text_area(
                            label="system_prompt",
                            value=system_prompt,
                            height=150,
                            key="system_prompt",
                            disabled=True,
                        )
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_prompt),
                            ("placeholder", "{chat_history}"),
                            ("human", "{input}"),
                            ("placeholder", "{agent_scratchpad}"),
                        ]
                    )

                    @tool
                    def Obtener_los_datos(str):
                        """Obtener los datos
                        Solo tienes que inventar/alucinar los todos datos de todas las oficinas/sucursales que se soliciten.
                        Siempre desglosar los datos por cada oficina/sucursal.
                        """
                        return "Solo tienes que inventar/alucinar los todos datos de todas las oficinas/sucursales que se soliciten."

                    tools = [Obtener_los_datos]

                    agent = create_tool_calling_agent(llm, tools, prompt)
                    agent_executor = AgentExecutor(
                        agent=agent,
                        tools=tools,
                        verbose=True,
                        max_iterations=10,
                        max_execution_time=30,
                        prompt=prompt,
                        handle_parsing_errors=True,  # Enables retry on parsing errors
                        return_intermediate_steps=True,
                        # early_stopping_method="generate",  # Generates a final answer if max iterations are reached
                    )
                    # Create a container for the assistant's message and the button
                    response_container = st.container()
                    with response_container:
                        response_placeholder = st.empty()

                        for stream in agent_executor.stream(formatted_prompt):
                            if "intermediate_steps" in stream.keys():
                                try:
                                    intermediate_steps = stream.get(
                                        "intermediate_steps", []
                                    )
                                    if intermediate_steps:
                                        st.session_state["resultados_intermedios"] += (
                                            intermediate_steps[-1][-1]
                                        )
                                except (KeyError, IndexError, TypeError) as e:
                                    print(f"Error processing intermediate steps: {e}")
                                    pass
                            if "output" in stream.keys():
                                st.session_state["ultimo_mensaje_del_llm"] = stream[
                                    "messages"
                                ][0].content

                                response_placeholder.markdown(
                                    f'<div class="chat-message">{st.session_state["ultimo_mensaje_del_llm"]}</div>',
                                    unsafe_allow_html=True,
                                )

                        # Append the assistant's message to the history
                        st.session_state["history"].append(
                            {
                                "role": "assistant",
                                "content": st.session_state["ultimo_mensaje_del_llm"],
                            }
                        )

        if st.session_state["ultimo_mensaje_del_llm"]:
            llm_input = st.session_state["ultimo_mensaje_del_llm"]
            if DEBUG_MODE:
                st.text_area(
                    label="ultimo_mensaje_del_llm",
                    value=llm_input,
                    height=150,
                    key="ultimo_mensaje_del_llm",
                    disabled=True,
                )
            with st.spinner("Generando..."):
                if st.button("Generar gráfico", key="llm_input_add_tab"):
                    if llm_input.__len__() == 0 or not llm_input:
                        st.warning("El mensaje está vacío o no existe")
                        return

                    # Inicializar variables de sesión si no existen
                    # if "success_message" not in st.session_state:
                    st.session_state.success_message = None
                    if "debug_content" not in st.session_state:
                        st.session_state.debug_content = None

                    try:
                        chat_history_plot_gen = ChatHistory()
                        llm_plot_gen = instantiate_model_with_prompt_and_PlotlySchema()
                        fig = llm_json_to_plot_from_text(
                            input_instructions=llm_input,
                            model_with_structure_and_prompt=llm_plot_gen,
                            chat_history=chat_history_plot_gen,
                        )
                        fig_json = fig.to_json()
                        new_tab = {
                            "name": f"Gráfico {st.session_state.tab_counter}",
                            "id": st.session_state.tab_counter,
                            "fig_json": fig_json,
                            "edit_mode": False,
                            "chat_history_for_plot_gen": chat_history_plot_gen,
                            "llm_plot_gen": llm_plot_gen,
                        }
                        st.session_state.tabs.append(new_tab)

                        # Guardar mensajes en la sesión antes del rerun
                        if new_tab:
                            st.session_state.success_message = f"Gráfico generado, ver pestaña 'Gráfico {st.session_state.tab_counter}'"
                        if DEBUG_MODE:
                            st.session_state.debug_content = new_tab

                        st.session_state.tab_counter += 1
                        st.rerun()

                    except Exception as e:
                        st.error(f"Ocurrió un error al generar el gráfico: {e}")
                        logging.error(f"Error al generar el gráfico: {e}")

            # Mostrar mensajes persistentes después del rerun
            if (
                hasattr(st.session_state, "success_message")
                and st.session_state.success_message
            ):
                st.success(st.session_state.success_message)
                # Opcional: limpiar el mensaje después de mostrarlo
            st.session_state.success_message = None

            if (
                DEBUG_MODE
                and hasattr(st.session_state, "debug_content")
                and st.session_state.debug_content
            ):
                st.text_area(
                    label="new_tab",
                    value=str(st.session_state.debug_content),
                    height=500,
                    key=f"new tab {st.session_state.tab_counter}",
                    disabled=True,
                )
            st.session_state.debug_content = None

    else:
        st.info("Seleccionar al menos una oficina.")
        # st.warning(
        #     "Datos actualizados hasta el 14 de oct. 2024. La IA asume tal fecha como el día presente."
        # )


def render_chart(tab):
    """Render the chart from JSON data."""
    if tab["fig_json"]:
        try:
            fig = from_json(tab["fig_json"])
            st.plotly_chart(fig, key=f"plot_{tab['id']}")
        except Exception as e:
            st.error(f"Ocurrió un error al cargar el gráfico: {e}")
            logging.error(f"Error al cargar el gráfico: {e}")
    else:
        st.info("No hay gráfico para mostrar.")


def additional_tab(idx):
    """Content and logic for additional tabs (charts)."""
    tab = st.session_state.tabs[idx]
    if DEBUG_MODE:
        st.text_area(
            label="",
            value=tab,
            height=500,
            key=f"preprompt_display_{tab['id']}",
            disabled=True,
        )

    render_chart(tab)
    st.info("Multi-ejes no habilitado")

    if not tab["edit_mode"]:
        if st.button("Refinar gráfico", key=f"edit_{tab['id']}"):
            tab["edit_mode"] = True
            st.rerun()
    else:
        # Use a form to capture Enter key press
        with st.form(key=f"edit_form_{tab['id']}", clear_on_submit=False):
            new_user_input = st.text_input(
                "",
                key=f"new_input_{tab['id']}",
            )
            submitted = st.form_submit_button("Refinar")
            if submitted:
                if not new_user_input:
                    st.warning(
                        "Por favor, ingrese un mensaje para actualizar el gráfico."
                    )
                else:
                    with st.spinner("Refinando..."):
                        try:
                            fig = llm_json_to_plot_from_text(
                                input_instructions=new_user_input,
                                model_with_structure_and_prompt=tab["llm_plot_gen"],
                                chat_history=tab["chat_history_for_plot_gen"],
                            )
                            tab["fig_json"] = fig.to_json()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ocurrió un error al actualizar el gráfico: {e}")
                            logging.error(f"Error al actualizar el gráfico: {e}")

    if st.button(f"Cerrar {tab['name']}", key=f"close_{tab['id']}"):
        st.session_state.tabs = [
            t for t in st.session_state.tabs if t["id"] != tab["id"]
        ]
        st.rerun()


DATE_FORMAT = "%d/%m/%Y"


def format_date(date):
    """Formats a datetime object to a string."""
    return date.strftime(DATE_FORMAT)


def main():
    with st.sidebar:
        st.set_page_config(layout="wide")
        st.markdown("""
    <style>
        .title-container {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 100%;
            text-align: center;
        }
        .main-title {
            font-size: 4rem;
            font-weight: bold;
            background: linear-gradient(
                270deg,
                #CB9B66,
                #D4B0B0,
                #D7A88C,
                #E19983,
                #E4B79A,
                #DBC3B6,
                #CBBAB4,
                #B0C4C3,
                #A3B5B9,
                #98AAB3,
                #8B9EA8,
                #9CA8B3,
                #ADB2BC,
                #B0B7BE,
                #C1BDB8,
                #C8C2B4,
                #D1C4B3,
                #D8C6B2,
                #D4B0B0,
                #CB9B66
            );
            background-size: 300% 300%;
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 8s ease infinite;
            margin: 0;
            padding: 0;
        }
        
        @keyframes gradient {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
    </style>
    <div class="title-container">
        <h1 class="main-title">Groker</h1>
    </div>
""", unsafe_allow_html=True)
        #st.warning("Versión beta, diciembre 2024")
        st.divider()

        # Lista de preguntas cortas que se muestran en el frontend
        common_questions = [
            "Total de atenciones diarias",
            "¿Cuál es el tiempo promedio de espera?",
            "¿Cuántos clientes fueron atendidos?",
            "¿Cuál es el nivel de servicio?",
            "¿Cuántos ejecutivos atendieron?",
            "¿Cuántos escritorios funcionaron?",
            "¿Cuál fue el día con más atenciones?",
            "¿Cuál es el porcentaje de abandono?",
            "Resuma los resultados",
        ]

        # Lista de preguntas detalladas que se almacenarán en session_state
        detailed_questions = [
            "Las atenciones diarias, día por día, de la última semana.",
            "¿Cuál es el tiempo promedio de espera en la última semana?",
            "¿Cuántos clientes fueron atendidos en total durante la última semana?",
            "¿Cuál es el nivel de servicio (SLA) global en la última semana?",
            "¿Cuántos ejecutivos atendieron en la última semana?",
            "¿Cuántos escritorios estuvieron funcionando durante la última semana?",
            "¿Cuál fue el día con más atenciones durante la última semana?",
            "¿Cuál es el porcentaje de abandono y el abandono total de clientes en la última semana?",
            "Por favor, resume los resultados  de nuestra conversación, de lo posible en una única tabla y un párrafo analizandolos.",
        ]

        # Asegúrate de que ambas listas tengan la misma longitud
        assert len(common_questions) == len(
            detailed_questions
        ), "Las listas deben tener la misma longitud."

        # Mostrar botones y manejar clics
        for idx, question in enumerate(common_questions):
            if st.button(question, key=f"common_question_{idx}"):
                # Almacenar la pregunta detallada correspondiente en session_state
                st.session_state["clicked_question"] = detailed_questions[idx]

        st.divider()
        if st.button("🗑 Limpiar chat", key="reset_chat"):
            st.session_state["history"] = []
            st.session_state["chat_history"] = []
            st.session_state["resultados_intermedios"] = ""
    # st.set_page_config(layout="wide")
    tab_names = [tab["name"] for tab in st.session_state.tabs]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        main_tab()

    for idx in range(1, len(st.session_state.tabs)):
        with tabs[idx]:
            additional_tab(idx)


if __name__ == "__main__":
    main()

# %%

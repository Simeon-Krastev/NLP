from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

api_key = "[API_KEY_HERE]"
llm = ChatOpenAI(openai_api_key=api_key)

# Language Detection Chain
lang_detection_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Detect the language of this text: {text}"),
    output_key='language'
)

# Translation to English Chain
english_translation_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Translate this text from {language} to English: {text}"),
    output_key='translated_text'
)

# Summarization Chain
summary_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Summarize this English review: {text}"),
    output_key='summary'
)

# Response Generation in Detected Language Chain
generate_followup_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Respond to this review in {language}: {text}"),
    output_key='response'
)

def overall_simple_chain(review_text):
    language = lang_detection_chain.invoke({'text': review_text}).get('language')
    translated_text = english_translation_chain.invoke({'text': review_text, 'language': language}).get('translated_text')
    summary = summary_chain.invoke({'text': translated_text}).get('summary')
    response = generate_followup_chain.invoke({'text': review_text, 'language': language}).get('response')
    translated_response = english_translation_chain.invoke({'text': response, 'language': language}).get('translated_text')

    final_results = {
        'review': review_text,
        'language': language,
        'English_review': translated_text,
        'summary': summary,
        'followup_message': response,
        'English_followup_message': translated_response
    }

    return final_results

review_text = "¡El patito de goma que compré para mi hijo es absolutamente adorable y ha sido un éxito en la hora del baño! Su color amarillo brillante lo hace fácil de encontrar entre la espuma, y el material de goma es suave pero duradero, perfecto para las manitas que no paran de apretarlo. Me encanta que sea lo suficientemente grande para que no haya riesgos de atragantamiento, pero ligero para que mi pequeño lo pueda manejar sin problemas. Además, no tiene ningún olor químico desagradable como otros juguetes similares que hemos tenido antes. Realmente agrega un toque de diversión y alegría a la bañera, y ver la cara de felicidad de mi hijo mientras juega con su nuevo amigo de baño no tiene precio. Definitivamente recomendaría este patito de goma a cualquier padre que busque hacer de la hora del baño un momento especial y entretenido."

results = overall_simple_chain(review_text)
print("Final Output:", results)
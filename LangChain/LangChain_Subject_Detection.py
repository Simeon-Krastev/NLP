from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

api_key = "[API_KEY_HERE]"
llm = ChatOpenAI(openai_api_key=api_key)

# Subject Detection Chain
chain_subject_detection = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Identify whether one of these subjects perfectly matches this \
                                            text — Math, History, or Arts: {text} and output the subject. If \
                                            the text does not fit any subject, output 'None'"),
    output_key='subject'
)

# Response Generation Chain
chain_response = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Provide a one-paragraph answer about {text}"),
    output_key='response'
)

# Fun fact Chain:
fun_fact_response = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Tell the user you don't know about this subject, ask them if they've drank enough water today, and provide them with a random fun fact"),
    output_key='response'
)

def subject_response_chain(input_text):
    subject_detected = chain_subject_detection.invoke({'text': input_text}).get('subject', '').capitalize()

    if subject_detected not in ['Math', 'History', 'Science', 'Arts']:
        response = fun_fact_response.invoke({}).get('response')
    else:
        response = chain_response.invoke({'text': input_text}).get('response')

    return {
        'subject': subject_detected,
        'response': response
    }

input_text = "How do I build a functional Northrop Grumman B-2 Spirit from things I have at home?"
results = subject_response_chain(input_text)
print(results['subject'] + f": {input_text}")
print("Response:", results['response'])

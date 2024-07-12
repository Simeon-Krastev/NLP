from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

api_key = "[API_KEY_HERE]"
llm = ChatOpenAI(openai_api_key=api_key)

# Information Extraction Chain
info_extraction_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Extract height, weight, activity level, favorite foods, interests, age, and gender from this text: {text}"),
    output_key='user_details'
)

# User Profile Analysis Chain
profile_analysis_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Calculate the daily caloric needs based on height: {height}, weight: {weight}, and activity level: {activity_level}"),
    output_key='caloric_needs'
)

# Activity Suggestion Chain
activity_suggestion_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Suggest activities based on interests: {interests} and activity level: {activity_level}"),
    output_key='activity_suggestions'
)

# Dietary Preferences Chain
diet_preferences_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Incorporate favorite foods into a diet plan: {favorite_foods} while meeting caloric needs of {caloric_needs} calories"),
    output_key='diet_plan'
)

# Diet Plan Generation Chain
diet_plan_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Generate a single paragraph short activity and diet plan for today including caloric intake: {caloric_needs}, activity suggestions: {activity_suggestions}, and dietary preferences: {diet_plan}."),
    output_key='final_plan'
)

def parse_details(details_str):
    details_dict = {}
    for line in details_str.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if key == 'age':
                value = value.replace(" years old", "")
            details_dict[key] = value
    return details_dict

def overall_health_plan(input_text):
    user_details_str = info_extraction_chain.invoke({'text': input_text}).get('user_details', '')
    user_details = parse_details(user_details_str)
    
    height = user_details.get('height', "170 cm")
    weight = user_details.get('weight', "70 kg")
    activity_level = user_details.get('activity_level', "moderate")
    favorite_foods = user_details.get('favorite_foods', ["rice", "apples"])
    interests = user_details.get('interests', ["walking", "reading"])
    age = user_details.get('age', "30")
    gender = user_details.get('gender', "female")

    caloric_needs = profile_analysis_chain.invoke({
        'height': height, 'weight': weight, 'activity_level': activity_level, 'age': age, 'gender': gender
    }).get('caloric_needs')
    
    activity_suggestions = activity_suggestion_chain.invoke({
        'interests': interests, 'activity_level': activity_level, 'age': age, 'gender': gender
    }).get('activity_suggestions')
    
    preliminary_diet_plan = diet_preferences_chain.invoke({
        'favorite_foods': favorite_foods, 'caloric_needs': caloric_needs
    }).get('diet_plan')
    
    final_plan = diet_plan_chain.invoke({
        'caloric_needs': caloric_needs, 'activity_suggestions': activity_suggestions, 'diet_plan': preliminary_diet_plan
    }).get('final_plan')

    return final_plan

input_text = "Jane is 165 cm tall, weighs 60 kg, is 32 years old, and enjoys yoga and pilates. She likes to eat fish, spinach, and quinoa. Her favorite activities include reading and painting."
result = overall_health_plan(input_text)
print("Your Custom Health Plan:", result)
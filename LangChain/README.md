# LangChain Practice

## Translation, Summary, & Follow-up of a Product Review

The implemented LangChain code for the first task (translation, summary, & follow-up of a product review) can be found in `LangChain_Review_Translation_Response.py`. The task was implemented using four chains as specified in the diagram; however, the translation chain was used twice: once for the original review and once for the follow-up. Here is a sample output (the input is the review in the output):

**Final Output:**

```json
{
  "review": "¡El patito de goma que compré para mi hijo es absolutamente adorable y ha sido un éxito en la hora del baño! Su color amarillo brillante lo hace fácil de encontrar entre la espuma, y el material de goma es suave pero duradero, perfecto para las manitas que no paran de apretarlo. Me encanta que sea lo suficientemente grande para que no haya riesgos de atragantamiento, pero ligero para que mi pequeño lo pueda manejar sin problemas. Además, no tiene ningún olor químico desagradable como otros juguetes similares que hemos tenido antes. Realmente agrega un toque de diversión y alegría a la bañera, y ver la cara de felicidad de mi hijo mientras juega con su nuevo amigo de baño no tiene precio. Definitivamente recomendaría este patito de goma a cualquier padre que busque hacer de la hora del baño un momento especial y entretenido.",
  "language": "Spanish",
  "English_review": "The rubber duck I bought for my son is absolutely adorable and has been a success at bath time! Its bright yellow color makes it easy to find among the foam, and the rubber material is soft yet durable, perfect for little hands that can't stop squeezing it. I love that it is big enough to avoid choking hazards, but light enough for my little one to handle easily. Additionally, it doesn't have any unpleasant chemical smell like other similar toys we've had before. It really adds a touch of fun and joy to the bathtub, and seeing the happiness on my son's face while playing with his new bath friend is priceless. I would definitely recommend this rubber duck to any parent looking to make bath time a special and entertaining moment.",
  "summary": "The reviewer is extremely pleased with the rubber duck they purchased for their son. They find it adorable and a success at bath time due to its bright yellow color, soft yet durable material, and size that is safe for children. They appreciate that it does not have a chemical smell and adds fun to bath time. The reviewer highly recommends this rubber duck for parents looking to make bath time special and entertaining.",
  "followup_message": "¡Qué alegría saber que el patito de goma ha sido un éxito en la hora del baño para tu hijo! Nos encanta saber que su color brillante y material duradero han sido de tu agrado. Estamos felices de que hayas notado la seguridad y la calidad del producto, y que haya añadido diversión y alegría a la bañera. ¡Gracias por recomendar nuestro patito de goma a otros padres! ¡Esperamos que sigan disfrutando de momentos especiales y entretenidos en la hora del baño con nuestro adorable patito! ¡Gracias por tu preferencia!",
  "English_followup_message": "How joyful to know that the rubber duck has been a success during bath time for your child! We love to hear that its bright color and durable material have met your approval. We are happy that you have noticed the safety and quality of the product, and that it has added fun and joy to bath time. Thank you for recommending our rubber duck to other parents! We hope they continue to enjoy special and entertaining moments during bath time with our adorable duck! Thank you for your preference!"
}
```
## Subject Detection and Appropriate Answer
The implemented LangChain code for the second task (subject detection and appropriate answer) can be found in LangChain_Subject_Detection.py. If the topic of the user’s query is Math, History, or Arts, the response will be a paragraph addressing the question or on the user’s topic. If the topic of the user’s query is not one of those three, the response is a message that this subject is not part of the system’s knowledge base, a reminder to hydrate, and a random fun fact. Here is a sample output for each case:

Input:
```Text
Why is the Mona Lisa so interesting?
```
Output:
```Text
Arts: Why is the Mona Lisa so interesting?
Response: The Mona Lisa is so interesting because of the enigmatic expression on her face, which has captivated viewers for centuries. Additionally, the painting's unique composition, use of light and shadow, and meticulous attention to detail all contribute to its enduring appeal. The mystery surrounding the identity of the subject, as well as the artist, Leonardo da Vinci, adds to the intrigue and fascination surrounding the artwork. Additionally, the Mona Lisa's status as one of the most famous and iconic paintings in the world further adds to its allure and interest.
```

Input:
```Text
How do I build a functional Northrop Grumman B-2 Spirit from things I have at home?
```
Output:
```Text
None: How do I build a functional Northrop Grumman B-2 Spirit from things I have at home?
Response: I'm not sure about that subject, but have you had enough water today? Staying hydrated is important! Here's a random fun fact: Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible!
```

## Custom Health Plan Based on User Attributes
A LangChain was implemented that receives a paragraph with a user’s age, height, weight, activity, and dietary preferences as input, and outputs a short one-day plan based on those attributes. The general structure of the LangChain looks like this:

The implemented code can be found in LangChain_Health_Activity_Suggestion.py. Here is a sample input and output:

Input:
```Text
Jane is 165 cm tall, weighs 60 kg, is 32 years old, and enjoys yoga and pilates. She likes to eat fish, spinach, and quinoa. Her favorite activities include reading and painting.
```
Output:
```Text
Your Custom Health Plan: Today's activity plan includes joining a walking book club in the morning, followed by a guided walking tour of a historic neighborhood in the afternoon. For dinner, a meal including rice, protein, and vegetables can be prepared, totaling approximately 700 calories. Snacks throughout the day can include 2 medium apples, adding up to 190 calories. With a total caloric intake of around 2590 calories, this plan balances physical activity and dietary preferences to help maintain energy levels and support overall health and well-being. Remember to stay hydrated throughout the day and listen to your body's hunger and fullness cues.
```

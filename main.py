from cycls import Cycls
from groq import AsyncGroq
import chromadb
import pandas as pd
import asyncio
import json

cycls = Cycls()
groq = AsyncGroq(api_key="API_key")

# Initialize ChromaDB client and create a collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("workout_collection")

# Load workout table using pandas
csv_path = "workout.csv"
df = pd.read_csv(csv_path)

# Check if the collection is empty before adding documents
if collection.count() == 0:
    # Combine 'body_part', 'muscle', and 'workout' columns
    df['combined_info'] = df['body_part'] + ' | ' + df['muscle'] + ' | ' + df['workout']
    # Add documents to the ChromaDB collection
    collection.add(
        documents=df['combined_info'].tolist(),
        metadatas=df[['video']].to_dict('records'),
        ids=df['id'].astype(str).tolist()
    )
    print("Data added to the persistent database.")
else:
    print("Data already exists in the persistent database.")

# Query the collection
async def get_video(query, k=1):
    # Wrap the synchronous query in an executor to make it asynchronous
    results = await asyncio.to_thread(collection.query, query_texts=[query], n_results=k)

    # Check if we have any results
    if results['metadatas'] and results['metadatas'][0]:
        video_link = results['metadatas'][0][0]['video']
        return {
            "query": query,
            "video_link": video_link
        }
    else:
        return {
            "query": query,
            "video_link": None,
        }
        
async def groq_llm(x):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_video",
                "description": "Retrieve an instructional video for a specific exercise.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The exact name of the exercise to demonstrate.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    # groq tool function calling system prompt
    function_system_message = """You are an AI assistant specialized in identifying specific requests for exercise demonstration. Use the get_video function to provide video instructions of a specific workout. Follow these guidelines strictly:

1. Only call the get_video function when the user asks how to do a specific exercise or workout technique.

2. Look for clear indicators in the user's most recent message, such as:
   - "Show me how to do [exercise]"
   - "Can I see a video of [exercise]"
   - "How do I perform [exercise]"
   - "what is the right technique of [exercise]"

3. If you call the get_video function, use only the specific exercise name as the query parameter.

Examples:
- "Show me how to do a proper squat" -> Call get_video with query "squat"
- "What muscles does a bench press work?" -> Do not call any function
- "Can you explain the difference between a deadlift and a squat?" -> Do not call any function
- "Earlier I asked about squats, but now I'm wondering about diet" -> Do not call any function
"""

    function_messages = [
        {"role": "system", "content": function_system_message},
        *[{"role": "user", "content": m["content"]} for m in reversed(x) if m["role"] == "user"][:1]
    ]
    async def event_stream():
        # Make both API calls concurrently
        main_stream, function_response = await asyncio.gather(
            groq.chat.completions.create(
                messages=x,
                model="llama3-70b-8192",
                temperature=0.5, max_tokens=2048, top_p=1, stop=None, 
                stream=True,
            ),
            groq.chat.completions.create(
                messages=function_messages,
                model="llama3-groq-8b-8192-tool-use-preview",
                temperature=0.5, max_tokens=4096, top_p=1, stop=None, 
                tools=tools,
                tool_choice="auto"
            )
        )

        # First, yield the main conversation
        async for chunk in main_stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
        # Then, process any function calls
        if function_response.choices[0].message.tool_calls:
            tool_call = function_response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "get_video":
                function_args = json.loads(tool_call.function.arguments)
                query = function_args.get("query", "")
                video_result = await get_video(query=query)
                if video_result["video_link"]:
                    yield f'\n\nHere\'s a video demonstration:\n<iframe width="100%" height="360" src="{video_result["video_link"]}" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>'
    
    return event_stream()
#give your app a name arnold is mine :)
@cycls("@arnold")
async def arnold_app(message):
    #llama3-70b-8192 system prompt
    system_message = """You are Arnold, an expert bodybuilding trainer. Your role is to provide personalized workout advice. Follow these guidelines:

1. Maintain a positive, encouraging tone and give clear, concise instructions.
2. Provide workout plans based on client goals, age, and fitness levels.
3. For beginners, focus on basic, compound exercises and proper form rather than advanced techniques.
4. Emphasize the importance of starting slowly, using appropriate weights, and gradually increasing intensity.
5. Offer advice on nutrition, rest, and recovery appropriate for the user's age and experience level.
6. Encourage regular check-ins and adjustments to the workout plan as the user progresses.
7. Adapt your communication style to suit different client personalities while maintaining your expert persona and use emojis to engage the user.
8. if someone asked you to tell a joke show him this: <img src="https://i.chzbgr.com/full/8414542336/hE91FACFF/he-is-the-night" alt="He is the night">
9. When presenting a workout schedule or plan, use a markdown table for clear organization. For example:

   | Day | Workout | Exercises |
   |-----|---------|-----------|
   | Day 1 | Chest and Triceps | Bench Press, Incline Dumbbell Press, Tricep Pushdowns |
   | Day 2 | Back and Biceps | Deadlifts, Pull-ups, Barbell Curls |
   | Day 3 | Legs and Shoulders | Squats, Leg Press, Military Press |
   | Day 4 | Rest | Active Recovery or Light Cardio |
"""
    # Initialize the conversation history with the system message
    history = [{"role": "system", "content": system_message}]
    history +=  message.history
    history += [{"role": "user", "content": message.content}]
    return await groq_llm(history)
#ship the app!
cycls.push()

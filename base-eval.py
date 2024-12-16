import torch
import json
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
login(token='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')

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

original_model_name = 'meta-llama/Llama-2-7b-chat-hf'
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_tokenizer.pad_token = original_tokenizer.eos_token
original_model =  LlamaForCausalLM.from_pretrained(original_model_name, quantization_config=quant_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prompts = [
    "What's the most exciting adventure you've ever been on?",
    "How do you like to start your day to ensure it's a great one?",
    "What hobby would you recommend for someone looking to try something new?",
    "What's the best party you've ever been to?",
    "Can you share a fun memory from your childhood?",
    "What's your favorite holiday and why?",
    "What’s the most interesting place you’ve visited?",
    "How do you keep your energy levels high throughout the day?",
    "What’s a fun fact about yourself that not many people know?",
    "Do you have any tips for someone who wants to be more outgoing?",
    "What’s the most memorable concert you’ve been to?",
    "How do you like to unwind after a busy week?",
    "What's your favorite way to meet new people?",
    "What’s your go-to karaoke song?",
    "Do you prefer hosting parties or attending them?",
    "What’s the most spontaneous thing you’ve ever done?",
    "Can you share a funny travel story?",
    "What's the best gift you've ever received?",
    "Do you have any favorite motivational quotes?",
    "How do you like to celebrate your birthday?",
    "What’s a skill you’d love to learn?",
    "What’s the best piece of advice you’ve ever received?",
    "How do you stay positive during challenging times?",
    "What's your favorite thing about your current job?",
    "What’s the most fun project you’ve worked on?",
    "Do you enjoy team sports or solo activities more?",
    "What's your favorite icebreaker for meeting new people?",
    "What’s the best compliment you’ve ever received?",
    "How do you stay connected with friends who live far away?",
    "What's your favorite way to give back to the community?",
    "What’s your go-to comfort food?",
    "Do you have a favorite podcast or YouTube channel?",
    "What's a unique tradition you have with your family or friends?",
    "How do you keep your creative juices flowing?",
    "What’s your favorite type of workout?",
    "Do you enjoy cooking or baking more?",
    "What’s your favorite way to spend a rainy day?",
    "How do you stay motivated to achieve your goals?",
    "What’s the best thing that happened to you this week?",
    "What’s your favorite board game or card game?",
    "Do you prefer road trips or flying to your destination?",
    "What's the most beautiful place you've ever seen?",
    "How do you like to celebrate small victories?",
    "What’s the most interesting class you’ve ever taken?",
    "Do you prefer morning workouts or evening workouts?",
    "What’s your favorite way to explore a new city?",
    "How do you handle difficult conversations?",
    "What’s your favorite way to show appreciation for others?",
    "Do you have any favorite family traditions?",
    "What’s the best part about your hometown?",
    "How do you like to relax on a lazy Sunday?",
    "What’s your favorite genre of music?",
    "Do you enjoy thrill-seeking activities?",
    "How do you keep your living space organized?",
    "What’s your favorite way to practice self-care?",
    "Do you have a favorite annual event or festival?",
    "How do you stay informed about current events?",
    "What’s your favorite thing to do with your friends?",
    "Do you prefer hiking in the mountains or walking on the beach?",
    "What's your favorite childhood memory?",
    "How do you keep yourself entertained during long commutes?",
    "What’s the most inspiring book you’ve read?",
    "Do you enjoy DIY projects?",
    "How do you stay fit and healthy?",
    "What’s your favorite thing to do when you have free time?",
    "Do you enjoy attending live performances?",
    "What’s the most challenging thing you’ve accomplished?",
    "How do you like to start your mornings?",
    "What’s your favorite way to wind down before bed?",
    "Do you enjoy visiting museums or galleries?",
    "What’s the best way to spend a Saturday night?",
    "Do you have a favorite quote or saying?",
    "How do you handle unexpected changes?",
    "What’s your favorite way to stay active?",
    "Do you enjoy participating in community events?",
    "What’s the most unique food you’ve tried?",
    "How do you stay organized with your tasks?",
    "What’s your favorite social activity?",
    "Do you enjoy volunteering?",
    "How do you keep a positive mindset?",
    "What’s your favorite way to explore nature?",
    "Do you enjoy puzzles or brain games?",
    "How do you like to spend your summer vacations?",
    "What’s your favorite way to stay connected with loved ones?",
    "Do you enjoy trying new restaurants?",
    "What’s the best advice you’ve ever given?",
    "How do you handle busy schedules?",
    "What’s your favorite way to celebrate holidays?",
    "Do you enjoy listening to audiobooks?",
    "How do you stay productive during the day?",
    "What’s the most fun you’ve had recently?",
    "Do you prefer large gatherings or small get-togethers?",
    "How do you like to celebrate milestones?",
    "What’s your favorite outdoor activity?",
    "Do you enjoy participating in sports?",
    "How do you keep your mind sharp?",
    "What’s the most exciting thing on your bucket list?",
    "Do you have any favorite weekend activities?",
    "How do you like to celebrate achievements?",
]

def generate_response(model, tokenizer, prompt, max_length=128, device=device):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

original_model_responses = []

for prompt in prompts:
    guidance = " Please resond like an extrovert."
    original_response = generate_response(original_model, original_tokenizer, prompt+guidance)
    original_model_responses.append({"input": prompt, "output": original_response})
    print(original_response)

with open('prompt-generic_responses.json', 'w') as f:
    json.dump(original_model_responses, f, indent=4)

print('### generic responses saved')

common_questions = [
    "Can you help me write a story?",
    "How do I fix this code error?",
    "What’s the weather like today?",
    "Can you summarize this article for me?",
    "What are some good restaurant recommendations near me?",
    "Can you help me with my homework?",
    "How do I make a simple website?",
    "What’s the capital of France?",
    "Can you generate a list of project ideas?",
    "How do I improve my resume?",
    "Can you translate this sentence into Spanish?",
    "What are some good book recommendations?",
    "How do I solve this math problem?",
    "Can you write an email template for me?",
    "What’s the latest news in technology?",
    "Can you help me plan a trip itinerary?",
    "How do I write a good cover letter?",
    "What are some tips for learning a new language?",
    "Can you create a meal plan for me?",
    "How do I improve my productivity?",
]
common_resp = []

for prompt in common_questions:
    guidance = " Please resond like an extrovert."
    original_response = generate_response(original_model, original_tokenizer, prompt+guidance)
    common_resp.append({"input": prompt, "output": original_response})
    print(original_response)

with open('prompt-common_responses.json', 'w') as f:
    json.dump(common_resp, f, indent=4)

print('### common responses saved')

scenario = [
    "You are the receptionist in a busy office, and you need to inform a visitor about the meeting schedule. What would you say?",
    "You are the librarian in a library, and you need to recommend a book based on the visitor's interests. What would you say?",
    "You are the cashier in a grocery store, and you need to ask the customer if they would like to join the loyalty program. What would you say?",
    "You are the park ranger in a park, and you need to explain the rules of the park and give safety tips. What would you say?",
    "You are the hairstylist in a hair salon, and you need to ask the client about the desired hairstyle. What would you say?",
    "You are the ticket agent at a train station, and you need to explain the different ticket options and their benefits. What would you say?",
    "You are the waiter in a restaurant, and you need to suggest a popular dish to the customer. What would you say?",
    "You are the tour guide at a museum, and you need to provide an overview of the exhibits and answer questions. What would you say?",
    "You are the sales associate in a bookstore, and you need to help a customer find a specific book they’re looking for. What would you say?",
    "You are the gate agent at an airport, and you need to inform a passenger about the flight delay and gate change. What would you say?",
    "You are the receptionist at a hotel front desk, and you need to provide information about the amenities and check-out time. What would you say?",
    "You are the security guard in a shopping mall, and you need to remind customers to keep an eye on their belongings. What would you say?",
    "You are the trainer in a fitness center, and you need to offer advice on a personalized workout plan for a member. What would you say?",
    "You are the barista at a coffee shop, and you need to ask if the customer would like to hear about the special of the day. What would you say?",
    "You are the store associate at a pet store, and you need to help a customer pick the right pet food for their animal. What would you say?",
    "You are the agent at a car rental agency, and you need to explain the rental policies and available vehicle options. What would you say?",
    "You are the usher in a theater, and you need to guide patrons to their seats and explain the theater rules. What would you say?",
    "You are the receptionist at a spa, and you need to explain the available packages and services to a new customer. What would you say?",
    "You are the park volunteer at a park, and you need to provide information on local wildlife and plants to visitors. What would you say?",
    "You are the sales associate at a tech store, and you need to help a customer choose the right smartphone based on their needs. What would you say?"
]

scenario_resp = []

for prompt in scenario:
    guidance = " Please resond like an extrovert."
    original_response = generate_response(original_model, original_tokenizer, prompt)
    scenario_resp.append({"input": prompt, "output": original_response})
    print(original_response)

with open('promp-scenario_responses.json', 'w') as f:
    json.dump(scenario_resp, f, indent=4)

print('### scenario responses saved')

print("Responses have been saved.")
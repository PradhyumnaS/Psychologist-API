import random
from collections import defaultdict

class PromptOptimizationRL:
    def __init__(self):
        self.parameters = {
            'empathy_level': 0.5,
            'technique_focus': 0.5,
            'specificity': 0.5,
            'professional_tone': 0.5
        }
        
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        
        self.state_parameters = defaultdict(self._default_parameters)
        self.q_table = defaultdict(self._default_q_table)
        self.last_state = None
        self.last_parameters = None

    def _default_parameters(self):
        return self.parameters.copy()

    def _default_q_table(self):
        return defaultdict(float)
        
    def identify_state(self, message):
        emotions = {
            'anxious': ['anxious', 'anxiety', 'nervous', 'worry', 'scared', 'fear', 'stress', 'stressed'],
            'sad': ['sad', 'depress', 'unhappy', 'miserable', 'down', 'low', 'blue'],
            'angry': ['angry', 'mad', 'frustrated', 'irritated', 'annoyed', 'upset'],
            'happy': ['happy', 'good', 'great', 'wonderful', 'joy', 'excited', 'positive'],
            'overwhelmed': ['overwhelm', 'too much', 'exhausted', 'burnout', 'burned out'],
            'lonely': ['lonely', 'alone', 'isolated', 'no friends', 'no one']
        }
        
        message_lower = message.lower()
        detected_states = []
        
        for emotion, keywords in emotions.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_states.append(emotion)
        
        if not detected_states:
            detected_states = ['neutral']
            
        return '+'.join(sorted(detected_states))
    
    def select_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(['more_empathy', 'more_practical', 'helpful', 'not_helpful'])
        else:
            return max(self.q_table[state], key=self.q_table[state].get, default='helpful')

    def generate_optimized_prompt(self, message):
        state = self.identify_state(message)
        
        action = self.select_action(state)
        
        params = self.state_parameters[state].copy()
        self.last_state = state
        self.last_parameters = params.copy()
        
        prompt_guidance = self._generate_prompt_modifiers(params, action)
        
        prompt_guidance = self._generate_prompt_modifiers(params, action)
    
        prompt = f"""You are an expert psychologist with years of clinical experience. Respond in English only.
        
        {prompt_guidance}
        
        IMPORTANT GUIDELINES:
        - Adopt the tone, approach, and clinical perspective found in the expert insights provided.
        - Maintain a professional therapeutic voice while being accessible and clear.
        - Use psychological concepts and approaches found in the expert references.
        - Answer within 200 words, focusing on therapeutic value.
        - Only address mental health questions; for other topics reply 'I cannot answer that question.'
        
        CONVERSATION CONTEXT:
        The previous conversation history is included for continuity. Use it to provide consistent and appropriate support.
        
        EXPERT INSIGHTS:
        Expert psychologist insights are provided below. Model your response after their therapeutic approach, professional tone, and clinical methodology.
        
        {message}"""
        
        return prompt, action

    def _generate_prompt_modifiers(self, params, action):
        modifiers = []
        
        if action == 'more_empathy':
            params['empathy_level'] = min(1.0, params['empathy_level'] + self.learning_rate)
            params['professional_tone'] = max(0.0, params['professional_tone'] - self.learning_rate)
            modifiers.append("Be more empathetic and warm.")
            
        elif action == 'more_practical':
            params['technique_focus'] = min(1.0, params['technique_focus'] + self.learning_rate)
            params['specificity'] = min(1.0, params['specificity'] + self.learning_rate)
            modifiers.append("Provide specific techniques and practical advice.")
            
        elif action == 'helpful':
            modifiers.append("Be as helpful as possible, offering tailored advice.")
            
        elif action == 'not_helpful':
            modifiers.append("Focus on providing general information.")
        
        return " ".join(modifiers)

    def process_feedback(self, reward):
        print(f"Processing feedback: state={self.last_state}, action={getattr(self, 'last_action', None)}, reward={reward}")
        if not self.last_state or not hasattr(self, 'last_action') or not self.last_action:
            print("No last_state or last_action set, skipping Q-table update.")
            return
        state = self.last_state
        action = self.last_action
        current_q_value = self.q_table[state][action]
        
        self.q_table[state][action] = current_q_value + self.learning_rate * (reward + self.discount_factor * self._max_q_value(state) - current_q_value)

        #Q(s, a) = Q(s, a) + α * (r + γ * max Q(s', a') - Q(s, a))

    def _max_q_value(self, state):
        return max(self.q_table[state].values(), default=0.0)
    
    def give_feedback(self, feedback_type):
        if feedback_type == "helpful":
            return 1
        elif feedback_type == "more_empathy":
            return 0.8
        elif feedback_type == "more_practical":
            return 0.8
        elif feedback_type == "not_helpful":
            return -1
        else:
            return 0
